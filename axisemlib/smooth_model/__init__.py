"""Generate smoothed radial models for the AxiSEM mesher."""

from pathlib import Path

import numpy as np

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


def write_tomography_model(
    profile_path: Path,
    output_path: Path,
    x_bounds: tuple[float, float] = (-270000.0, 270000.0),
    y_bounds: tuple[float, float] = (-300000.0, 300000.0),
    z_bounds: tuple[float, float] = (-220000.0, 0.0),
    shape: tuple[int, int, int] = (101, 121, 45),
) -> Path:
    """Interpolate a radial profile onto a SPECFEM tomography grid."""
    nx, ny, nz = shape
    if min(shape) < 2:
        raise ValueError("each grid dimension must contain at least two points")
    data = np.loadtxt(profile_path, ndmin=2)
    if data.shape[1] < 4:
        raise ValueError("profile must contain depth, vp, vs, and rho columns")
    order = np.argsort(data[:, 0])
    depth_m = data[order, 0] * 1000.0
    x = np.linspace(*x_bounds, nx)
    y = np.linspace(*y_bounds, ny)
    z = np.linspace(*z_bounds, nz)
    if z[0] < depth_m[0] or z[-1] > depth_m[-1]:
        raise ValueError("tomography depth range extends outside the radial profile")

    vp = np.interp(z, depth_m, data[order, 1])
    vs = np.interp(z, depth_m, data[order, 2])
    rho = np.interp(z, depth_m, data[order, 3])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        stream.write(f"{x[0]:f} {y[0]:f} {z[0]:f} {x[-1]:f} {y[-1]:f} {z[-1]:f}\n")
        stream.write(f"{x[1]-x[0]:f} {y[1]-y[0]:f} {z[1]-z[0]:f}\n")
        stream.write(f"{nx} {ny} {nz}\n")
        stream.write(
            f"{vp.min():f} {vp.max():f} {vs.min():f} {vs.max():f} "
            f"{rho.min():f} {rho.max():f}\n"
        )
        for iz, depth in enumerate(z):
            for north in y:
                for east in x:
                    stream.write(
                        f"{east:f} {north:f} {depth:f} "
                        f"{vp[iz]:f} {vs[iz]:f} {rho[iz]:f}\n"
                    )
    return output_path
