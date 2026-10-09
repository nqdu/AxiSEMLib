"""Generate smoothed radial models for the AxiSEM mesher."""

from pathlib import Path

from .smooth import smooth_pde


def generate_model(
    model_name: str = "prem",
    sigma_km: float = 5.0,
    ngll: int = 5,
    element_size_km: float = 1.0,
    output_dir: Path = Path("."),
    plot: bool = True,
) -> tuple[Path, Path]:
    """Smooth a 1D model and write its AxiSEM and depth-profile files."""
    if model_name not in ("prem", "ak135"):
        raise ValueError("model_name must be 'prem' or 'ak135'")
    if sigma_km <= 0 or element_size_km <= 0 or ngll < 2:
        raise ValueError("sigma_km and element_size_km must be positive; ngll must be at least 2")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, depths, locations, smoothed, core = smooth_pde(
        ngll, sigma_km, ("vp", "vs", "rho"), model_name, element_size_km
    )

    external_path = output_dir / f"{model_name}.smooth.bm"
    profile_path = output_dir / f"{model_name}.txt"
    with external_path.open("w") as stream:
        stream.write(f"NAME {model_name}_smooth\n")
        stream.write("ANELASTIC       T\nANISOTROPIC     T\nUNITS           m\n")
        stream.write(
            "COLUMNS       radius      rho      vpv      vsv      qka      qmu      vph      vsh      eta\n"
        )
        for depth, (vp, vs, rho) in zip(locations[:, 0], smoothed):
            radius = 1000.0 * (6371.0 - depth)
            stream.write(
                f"\t {radius:f} {rho:f} {vp:f} {vs:f} 9999.000000 9999.000000 "
                f"{vp:f} {vs:f} 1.000000\n"
            )
        stream.write(core)

    with profile_path.open("w") as stream:
        for depth, (vp, vs, rho) in zip(locations[:, 0], smoothed):
            stream.write(f"{-depth:f} {vp:f} {vs:f} {rho:f} \n")

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 14))
        ax.plot(smoothed[:, 1] / 1000.0, -locations[:, 0], label="smoothed Vs")
        ax.plot(model[:, :, 1].ravel() / 1000.0, -depths.ravel(), label="original Vs")
        ax.set(xlabel="Vs (km/s)", ylabel="Depth (km)")
        ax.legend()
        fig.savefig(output_dir / "smooth.jpg")
        plt.close(fig)

    return external_path, profile_path
