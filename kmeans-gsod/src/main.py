"""
Entry point for the GSOD climate clustering application.

Pipeline:
    list S3 -> sample stations -> download -> clean -> engineer features
    -> scale -> choose k -> fit K-Means -> profile -> validate -> plot -> export

Run from the project root:

    python -m src.main
    python -m src.main --stations 400 --year 2022
    python -m src.main --k 6              # override automatic k selection
    python -m src.main --offline          # reuse the cache, no network calls

In PyCharm, set the working directory to the project root and the script path to
src/main.py, or add a run configuration with module name `src.main`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import clustering, config, data_loader, features, visualise

logger = logging.getLogger("gsod_kmeans")


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster global weather stations by climate using K-Means "
        "and NOAA GSOD data from the AWS Registry of Open Data."
    )
    parser.add_argument("--year", type=int, default=config.YEAR,
                        help=f"observation year (default {config.YEAR})")
    parser.add_argument("--stations", type=int, default=config.MAX_STATIONS,
                        help=f"maximum stations to sample (default {config.MAX_STATIONS})")
    parser.add_argument("--per-block", type=int, default=config.STATIONS_PER_BLOCK,
                        help="stations per WMO block, controls global spread")
    parser.add_argument("--k", type=int, default=None,
                        help="force a specific k instead of using the inertia elbow")
    parser.add_argument("--offline", action="store_true",
                        help="use only cached files, make no S3 requests")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                        help="where figures and CSV results are written")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def display_path(path) -> str:
    """
    Render a path relative to the project root for logging.

    Absolute paths in logs leak the machine's directory layout and username, and
    the log is committed to the repository. A relative path is also simply easier
    to read. Falls back to the full path if it lies outside the project.
    """
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


# --------------------------------------------------------------------------- #
# Stage 1: acquire
# --------------------------------------------------------------------------- #
def acquire_station_files(args) -> list[Path]:
    """Return local paths to station CSVs, downloading them if necessary."""
    cache_dir = config.DATA_DIR / "gsod" / str(args.year)

    if args.offline:
        paths = sorted(cache_dir.glob("*.csv"))
        if not paths:
            raise data_loader.DataAcquisitionError(
                f"--offline was given but no cached files exist in {cache_dir}. "
                "Run once without --offline first."
            )
        logger.info("Offline mode: using %d cached station files", len(paths))
        return paths

    logger.info("Listing s3://%s/%d/ ...", config.S3_BUCKET, args.year)
    catalogue = data_loader.list_station_objects(args.year)

    sample = data_loader.select_global_sample(
        catalogue,
        stations_per_block=args.per_block,
        max_stations=args.stations,
    )

    logger.info("Downloading %d station files ...", len(sample))
    return data_loader.download_stations(sample["key"].tolist(), args.year)


# --------------------------------------------------------------------------- #
# Stage 2: clean and engineer
# --------------------------------------------------------------------------- #
def build_features(paths: list[Path]) -> pd.DataFrame:
    """Load every station file, clean it, and assemble the feature table."""
    station_frames = {}
    unreadable = 0

    for path in paths:
        try:
            station_frames[path.stem] = data_loader.load_station(path)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            unreadable += 1
            logger.warning("Could not parse %s: %s", path.name, exc)

    if unreadable:
        logger.info("Skipped %d unreadable files", unreadable)

    return features.build_feature_table(station_frames)


# --------------------------------------------------------------------------- #
# Stage 3: cluster
# --------------------------------------------------------------------------- #
def run_clustering(table: pd.DataFrame, forced_k: int | None):
    matrix, scaler = clustering.scale_features(table)

    diagnostics = clustering.evaluate_k_range(matrix)
    elbow_k = clustering.choose_k(diagnostics)
    silhouette_k = clustering.choose_k_by_silhouette(diagnostics)
    consensus_k = int(clustering.rank_diagnostics(diagnostics)["mean_rank"].idxmin())

    chosen_k = forced_k or elbow_k
    if forced_k:
        logger.info("Using k=%d as requested (the elbow suggested k=%d)",
                    forced_k, elbow_k)

    model, labels = clustering.fit_kmeans(matrix, chosen_k)
    selection = {
        "elbow_k": elbow_k,
        "silhouette_k": silhouette_k,
        "consensus_k": consensus_k,
    }
    return matrix, scaler, diagnostics, chosen_k, model, labels, selection


# --------------------------------------------------------------------------- #
# Stage 4: report
# --------------------------------------------------------------------------- #
def report(table, labels, profile, cluster_names, diagnostics, chosen_k, selection,
           representatives, fragility) -> None:
    """Print the findings in a form that can be lifted into the summary."""
    line = "=" * 78

    print(f"\n{line}\nK-SELECTION DIAGNOSTICS\n{line}")
    combined = diagnostics.join(clustering.rank_diagnostics(diagnostics))
    print(combined.round(4).to_string())

    spread = diagnostics["silhouette"].max() - diagnostics["silhouette"].min()
    print(f"\n  Silhouette range across all k : {spread:.4f}  "
          f"({diagnostics['silhouette'].min():.3f} to {diagnostics['silhouette'].max():.3f})")
    print(f"  Elbow of the inertia curve    : k = {selection['elbow_k']}   <- used")
    print(f"  Best silhouette alone         : k = {selection['silhouette_k']}")
    print(f"  Best mean rank of all three   : k = {selection['consensus_k']}")
    print(f"  Fitted                        : k = {chosen_k}")
    if spread < 0.05:
        print("\n  The silhouette curve is nearly flat, which is the expected result for"
              "\n  climate: it is a continuum, not a set of separated blobs. The maximum"
              "\n  of a flat curve is noise, so k comes from the elbow of the inertia"
              "\n  curve instead, which is the criterion that rewards parsimony.")

    print(f"\n{line}\nCLUSTER PROFILES\n{line}")
    for cluster, row in profile.iterrows():
        print(f"\nCluster {cluster} -- {cluster_names[cluster]}  "
              f"({int(row.n_stations)} stations)")
        print(f"  mean temperature      {row.temp_mean_c:7.1f} deg C")
        print(f"  seasonality           {row.temp_seasonality_c:7.1f} deg C")
        print(f"  diurnal range         {row.diurnal_range_c:7.1f} deg C")
        print(f"  dew point depression  {row.dewpoint_depression_c:7.1f} deg C")
        print(f"  annual precipitation  {row.precip_total_mm:7.0f} mm")
        print(f"  wet days              {row.wet_day_fraction:7.1%}")
        print(f"  snow days             {row.snow_day_fraction:7.1%}")
        print(f"  thunder days          {row.thunder_day_fraction:7.1%}")
        print(f"  -- withheld from model --")
        print(f"  mean |latitude|       {row.abs_latitude:7.1f} deg")
        print(f"  mean elevation        {row.elevation_m:7.0f} m")

        print("  most typical stations (nearest the centroid):")
        for name in representatives.get(cluster, []):
            print(f"    - {name.title()}")

    print(f"\n{line}\nVALIDATION AGAINST WITHHELD GEOGRAPHY\n{line}")
    for column, description in [
        ("abs_latitude", "absolute latitude"),
        ("elevation_m", "elevation"),
    ]:
        share = clustering.variance_explained(table, labels, column)
        print(f"  Cluster membership explains {share:6.1%} of the variance in {description}")
    print("\n  Neither latitude nor elevation was part of the feature set, so any"
          "\n  agreement is inferred from weather behaviour alone.")

    print(f"\n{line}\nHOW FIRM ARE THE BOUNDARIES?\n{line}")
    print(f"  {fragility:.1%} of stations sit almost as close to a neighbouring cluster"
          f"\n  as to their own. Climate is continuous, so a hard partition has to cut"
          f"\n  somewhere; these are the stations where the assignment is a close call"
          f"\n  and would move under a different random seed or a slightly wider sample.")


def export_results(table, labels, profile, diagnostics, cluster_names, output_dir: Path):
    """Write the labelled station table and supporting CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labelled = table.copy()
    labelled["cluster"] = labels
    labelled["cluster_label"] = [cluster_names[c] for c in labels]
    labelled.to_csv(output_dir / "station_clusters.csv")

    profile.assign(cluster_label=[cluster_names[c] for c in profile.index]).to_csv(
        output_dir / "cluster_profiles.csv"
    )
    diagnostics.to_csv(output_dir / "k_diagnostics.csv")

    logger.info("Wrote CSV results to %s", display_path(output_dir))


def main(argv=None) -> int:
    args = parse_arguments(argv)
    configure_logging(args.verbose)

    try:
        paths = acquire_station_files(args)
        table = build_features(paths)
        logger.info("Feature table ready: %d stations x %d features",
                    len(table), len(config.FEATURE_COLUMNS))

        matrix, scaler, diagnostics, chosen_k, model, labels, selection = run_clustering(
            table, args.k
        )

        profile = clustering.profile_clusters(table, labels)
        cluster_names = {
            cluster: features.describe_cluster(row) for cluster, row in profile.iterrows()
        }

        representatives = clustering.representative_stations(table, matrix, model, labels)
        fragility = clustering.boundary_fragility(matrix, model, labels)

        report(table, labels, profile, cluster_names, diagnostics, chosen_k, selection,
               representatives, fragility)

        visualise.plot_k_selection(diagnostics, chosen_k, args.output_dir)
        visualise.plot_world_map(table, labels, cluster_names, args.output_dir)
        visualise.plot_cluster_profiles(profile, cluster_names, args.output_dir)
        visualise.plot_feature_space(table, labels, cluster_names, args.output_dir)
        logger.info("Wrote 4 figures to %s", display_path(args.output_dir))

        export_results(table, labels, profile, diagnostics, cluster_names, args.output_dir)

    except (data_loader.DataAcquisitionError, features.InsufficientDataError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
