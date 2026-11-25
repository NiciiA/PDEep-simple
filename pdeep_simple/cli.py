"""Seven small argparse commands; run from simple/ or install with pip -e."""
import argparse
from .data import PDES
from .geometry import GEOMETRIES


def parser():
    p = argparse.ArgumentParser(description='PDEep/simple: numerical PDE cases and geometry-conditioned reconstruction')
    commands = p.add_subparsers(dest='command', required=True)
    data = commands.add_parser('data', help='Generate independent numerical physical cases')
    data.add_argument('--config', help='Optional TOML mathematical dataset settings')
    data.add_argument('--tier', choices=['T1', 'G1', 'T2', 'T3', 'T4'], default='G1')
    data.add_argument('--pdes', help='Comma-separated PDEs; default is all supported by the tier')
    data.add_argument('--geometries', default='square,disk,lshape', help='Comma-separated geometry names; T1 requires square')
    for split, default in (('train', 8), ('val', 2), ('test', 2)):
        data.add_argument('--' + split, type=int, default=default, help='Physical cases per PDE-geometry combination')
    data.add_argument('--resolution', type=int, help='Grid subdivisions (default 12, or config); rounded up to even')
    data.add_argument('--frames', type=int, help='Stored times including 0 and 1 (default 21, or config)')
    data.add_argument('--dt', type=float, help='Maximum time step (default .002, or config)')
    data.add_argument('--workers', default='auto', help='Positive integer or auto (at most four)')
    data.add_argument('--seed', type=int, default=42); data.add_argument('--out', required=True)
    inspect = commands.add_parser('inspect', help='Mesh, physical fields, snapshots, observations and summaries')
    inspect.add_argument('--data', required=True); inspect.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    inspect.add_argument('--case', type=int, default=0, help='Zero-based index in split')
    inspect.add_argument('--random', type=int, default=0, help='Select this many random cases instead of --case')
    inspect.add_argument('--seed', type=int, default=42); inspect.add_argument('--observations', type=int, default=32)
    inspect.add_argument('--noise', type=float, default=.01, help='Absolute Gaussian observation standard deviation')
    inspect.add_argument('--out', default='runs/inspection')
    train = commands.add_parser('train', help='Train A/C/D/E/F/G/H-Geo using independent physical cases')
    train.add_argument('--data', required=True); train.add_argument('--model', choices=['A', 'C', 'D', 'E', 'F', 'G', 'H'], default='A')
    train.add_argument('--size', choices=['s', 'm', 'l', 'xl', 'xxl'], default='s')
    for name in ('width', 'layers', 'heads', 'memory'):
        train.add_argument('--' + name, type=int, help='Manual preset override')
    train.add_argument('--fourier-bands', type=int, default=6, help='D/G-Geo decoder frequency bands')
    train.add_argument('--wave-modes', type=int, default=32, help='E-Geo low-frequency FEM wave modes')
    train.add_argument('--evolution-steps', type=int, default=32, help='F-Geo trajectory steps over the physical interval')
    train.add_argument('--siren-omega', type=float, default=30., help='G-Geo sinusoidal activation frequency')
    train.add_argument('--green-sources', type=int, default=64, help='H-Geo spatial quadrature samples')
    train.add_argument('--modes', default='forward,inverse,hybrid')
    train.add_argument('--loss', default='field=1,aux=.1', help='field,aux,kinematic,weak,initial,boundary weights, e.g. field=1,weak=.001')
    train.add_argument('--staged', action='store_true', help='Deterministic field-only training, then the selected --loss weights')
    train.add_argument('--stage-start', type=int, help='Last field-only update; default 20%% of total updates (at least one)')
    train.add_argument('--updates', type=int, default=3000, help='Total desired updates, including updates before --resume')
    train.add_argument('--batch', type=int, default=2, help='Physical cases per update, sequential gradient accumulation')
    train.add_argument('--queries', type=int, default=256, help='Fresh area-uniform continuous space-time queries per case')
    train.add_argument('--observations', default='0,8,32,128', help='Random count; inverse/hybrid training omit zero')
    train.add_argument('--noise', type=float, default=.01, help='Absolute observation standard deviation')
    train.add_argument('--lr', type=float, default=1e-4); train.add_argument('--validate-every', type=int, default=10)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--resume', help='last.pt checkpoint; restores saved training settings')
    train.add_argument('--geometries', help='Restrict training families, e.g. square,disk for an L-family holdout')
    train.add_argument('--flow', action='store_true', help='Staged field autoencoder then conditional latent flow matching')
    train.add_argument('--latent-dim', type=int, default=32)
    train.add_argument('--latent-points', type=int, default=256, help='Area-uniform reference field samples for the training-only encoder')
    train.add_argument('--ae-updates', type=int, help='First-stage updates; default min(500, total updates / 3)')
    train.add_argument('--flow-steps', type=int, default=32, help='Heun steps for validation samples')
    train.add_argument('--run', required=True)
    for name in ('eval', 'baseline'):
        sub = commands.add_parser(name, help='Held-out quadrature evaluation' if name == 'eval' else 'Prior, RBF, T1/G1 elliptic Bayes, T1 ridge, T2 scalar grid')
        sub.add_argument('--data', required=True); sub.add_argument('--split', choices=['train', 'val', 'test'], default='test')
        sub.add_argument('--modes', default='forward,inverse,hybrid')
        sub.add_argument('--observations', default='0,8,32,128', help='Zero explicitly tests the unobserved prior')
        sub.add_argument('--noise', type=float, default=.01); sub.add_argument('--seed', type=int, default=123)
        sub.add_argument('--frames', type=int, default=5, help='Stored times for quadrature and finite-difference diagnostics')
        sub.add_argument('--geometries', help='Restrict evaluation families, e.g. lshape')
        sub.add_argument('--save-cases', type=int, default=3, help='Saved prediction cases per PDE')
        sub.add_argument('--chunk', type=int, default=512, help='Neural query chunk size')
        sub.add_argument('--grids', help='Comma-separated bounding-box grid sizes, e.g. 128,256,512; default FEM nodes')
        sub.add_argument('--max-cases', type=int, help='Limit selected physical cases; grid evaluation defaults to at most eight')
        sub.add_argument('--pde', choices=PDES)
        sub.add_argument('--refine', type=int, default=0, help='Red-refinement levels for reference only; requires explicit --max-cases')
        sub.add_argument('--out', required=name == 'baseline')
        if name == 'eval':
            sub.add_argument('--run', required=True); sub.add_argument('--checkpoint', default='best.pt', choices=['best.pt', 'last.pt'])
        else:
            sub.add_argument('--workers', default='1', help='Positive process count or auto (at most four)')
            sub.add_argument('--memory-gb', type=float, default=2., help='Conservative total worker memory budget; auto also respects available RAM')
            sub.add_argument('--normalization-run', help="Use this run's fixed scales for a family-holdout comparison")
            sub.add_argument('--methods', help='prior,rbf,ridge,bayes,grid,pod,gp,map; default retains applicable tier references')
            sub.add_argument('--pod-rank', type=int, default=32, help='Maximum weighted POD rank; fitted on training only')
            sub.add_argument('--pod-ridge', type=float, default=1., help='Positive noise-weighted POD ridge penalty')
            sub.add_argument('--pod-model', help='Saved training representation .npz; default OUT/pod.npz, created if missing')
            sub.add_argument('--gp-lengthscale', type=float, default=.2, help='Spatial squared-exponential length in physical coordinates')
            sub.add_argument('--gp-amplitude', type=float, default=1., help='GP prior standard deviation in physical field units')
            sub.add_argument('--map-maxiter', type=int, default=100, help='L-BFGS-B iteration limit for at most 16 free parameters')
    for sub in (train, commands.choices['eval'], commands.choices['baseline']):
        sub.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
        sub.add_argument('--threads', type=int, default=2, help='PyTorch CPU threads')
    plot = commands.add_parser('plot', help='Loss curves, fields and cross-run comparisons')
    plot.add_argument('--run'); plot.add_argument('--what', default='losses,fields,comparisons')
    plot.add_argument('--compare', nargs='*', default=[], help='Other run/evaluation directories or metrics CSVs')
    plot.add_argument('--runs', nargs='+', help='Compare saved neural runs or baseline evaluation directories')
    plot.add_argument('--methods', help='Select methods when --runs includes a classical baseline directory')
    plot.add_argument('--metric', choices=['rmse', 'normalized_rmse'], default='rmse',
                      help='Error-versus-sensors/geometry/time metric; normalized comparisons need matching training scales')
    plot.add_argument('--data', help='Dataset for same-case prediction comparisons')
    plot.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    plot.add_argument('--pde', choices=PDES)
    plot.add_argument('--geometry', choices=GEOMETRIES)
    plot.add_argument('--case', type=int, default=0, help='Index among cases matching PDE/geometry')
    plot.add_argument('--modes', default='forward,inverse,hybrid')
    plot.add_argument('--grid', type=int, default=128)
    plot.add_argument('--observations', type=int, default=32)
    plot.add_argument('--noise', type=float, default=.01); plot.add_argument('--seed', type=int, default=123)
    plot.add_argument('--frames', type=int, default=3)
    plot.add_argument('--chunk', type=int, default=512)
    plot.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    plot.add_argument('--max-fields', type=int, default=9); plot.add_argument('--out')
    for sub in (commands.choices['eval'], plot):
        sub.add_argument('--samples', type=int, default=32, help='Whole-field flow samples, never independent query noise')
        sub.add_argument('--flow-steps', type=int, default=32, help='Heun flow ODE steps')
        sub.add_argument('--save-samples', type=int, default=4, help='Individual sample fields retained for plotting')
        sub.add_argument('--interval', type=float, default=.9, help='Empirical central interval probability')
    check = commands.add_parser('check', help='Analytical numerical checks and A/C backward smoke checks')
    check.add_argument('--threads', type=int, default=2); check.add_argument('--out', default='runs/checks.json')
    return p


def main():
    p = parser(); args = p.parse_args()
    try:
        if args.command == 'data':
            from .data import generate
            generate(args)
        elif args.command == 'inspect':
            from .plotting import inspect_data
            inspect_data(args)
        elif args.command == 'train':
            from .training import train
            train(args)
        elif args.command in ('eval', 'baseline'):
            from .evaluation import evaluate
            evaluate(args, args.command == 'baseline')
        elif args.command == 'plot':
            from .plotting import plot
            plot(args)
        else:
            from .checks import run_checks
            run_checks(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
