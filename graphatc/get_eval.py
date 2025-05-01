import argparse
import graphatc
from graphatc.metric import metric_util
PROJECT_PATH = graphatc.__path__[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_NAME", type=str, default=None)
    parser.add_argument("--LEVEL", type=str, default="1")
    args = parser.parse_args()

    if args.EXP_NAME is not None:
        logdir = PROJECT_PATH + f"/log/{args.EXP_NAME}_L{args.LEVEL}"
        df = metric_util.concat_log(logdir, logdir+"/concat.csv", True)
        print(f"Log dir: {logdir}")
        print(f"Length: {len(df)}")
        metric_df = metric_util.get_five_metric_distinguish(df, level=args.LEVEL)
        print(metric_df)
        metric_df.to_csv(logdir + "/eval.csv", index=False)
        print(f"Eval results saved to {logdir}/eval.csv")