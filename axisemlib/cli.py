"""Command-line interface for the AxiSEMLib workflows."""

import argparse
from pathlib import Path
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axisemlib", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("merge", help="Build AxiSEM boundary_faces.dat")
    merge.add_argument("system", choices=("cube2sph", "cart"))
    merge.add_argument("input_dir", type=Path, help="SPECFEM DATABASES_MPI directory")
    merge.add_argument("output_file", type=Path, help="AxiSEM boundary_faces.dat path")
    merge.add_argument("--utm-zone", type=int, help="UTM zone required for cart input")

    seismogram = commands.add_parser(
        "seismogram", help="Sum and rotate recorded seismograms to NEZ"
    )
    seismogram.add_argument("run_dir", type=Path, help="AxiSEM run directory")
    seismogram.add_argument("--output-dir", type=Path, default=Path("SEISMOGRAMS"))

    coupling = commands.add_parser("coupling", help="Generate SPECFEM coupling files")
    coupling.add_argument("config", type=Path, help="Coupling YAML configuration")

    transpose = commands.add_parser("transpose", help="Transpose NetCDF wavefields")
    transpose.add_argument("size_gb_per_rank", type=float, help="Read buffer size in GiB")
    transpose.add_argument("run_dirs", type=Path, nargs="+", help="AxiSEM run directories")

    model = commands.add_parser("model", help="Generate a smoothed AxiSEM model")
    model.add_argument("--model", choices=("prem", "ak135"), default="prem")
    model.add_argument("--sigma-km", type=float, default=5.0)
    model.add_argument("--ngll", type=int, default=5)
    model.add_argument("--element-size-km", type=float, default=1.0)
    model.add_argument("--output-dir", type=Path, default=Path("."))
    model.add_argument("--no-plot", action="store_true")

    prepare = commands.add_parser(
        "prepare", help="Prepare one AxiSEM injection event",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from .prepare_axisem_injection import add_arguments

    add_arguments(prepare)
    prepare.set_defaults(_prepare_parser=prepare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "merge":
        if args.system == "cart" and args.utm_zone is None:
            parser.error("merge cart requires --utm-zone")
        if args.system == "cube2sph" and args.utm_zone is not None:
            parser.error("--utm-zone applies only to merge cart")
        from .surface_merge import merge_surfaces

        sources = merge_surfaces(
            args.input_dir, args.output_file, args.system, args.utm_zone
        )
        print(f"Wrote {len(sources)} faces to {args.output_file}")
        return 0

    if args.command == "seismogram":
        from .postprocess_seismograms import process

        process(args.run_dir.resolve(), args.output_dir)
        return 0

    if args.command == "transpose":
        if args.size_gb_per_rank <= 0:
            parser.error("transpose requires a positive size_gb_per_rank")
        from .transpose_fields import main as transpose_fields

        transpose_fields(
            [str(args.size_gb_per_rank), *(str(path) for path in args.run_dirs)]
        )
        return 0

    if args.command == "model":
        from .smooth_model import generate_model

        try:
            external, profile = generate_model(
                args.model, args.sigma_km, args.ngll,
                args.element_size_km, args.output_dir, not args.no_plot,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote {external} and {profile}")
        return 0

    if args.command == "prepare":
        from .prepare_axisem_injection import prepare_inputs

        prepare_inputs(args, args._prepare_parser)
        return 0

    from .run_coupling import main as run_coupling

    return run_coupling(args.config)
