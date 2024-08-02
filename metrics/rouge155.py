import os
import re
from pathlib import Path
import tempfile
import subprocess


def rouge_reference_base(
    refs: list[str],
    hyps: list[str],
    args: str = "-n 2",
) -> str:
    """

    Run the reference Perl ROUGE script, returning its exact outputs string.

    Other functions (rouge_reference_individual and rouge_reference_overall) use
    regular expressions to search this output string and return scores. This
    function should be be run directly.

    """

    rouge_path = "metrics/ROUGE-1.5.5"
    # rouge_path = "data/libraries/ROUGE-1.5.5"

    temp = tempfile.TemporaryDirectory()

    ref_dir = Path(temp.name) / "ref"
    hyp_dir = Path(temp.name) / "hyp"

    ref_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)

    conf_file = Path(temp.name) / "config.xml"

    conf_entries = []

    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        ref_path = ref_dir / f"{i}.spl"
        hyp_path = hyp_dir / f"{i}.spl"

        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        conf_entries.append(
            f"""
            <EVAL ID="{i}">
                <PEER-ROOT>{hyp_dir}</PEER-ROOT>
                <MODEL-ROOT>{ref_dir}</MODEL-ROOT>
                <INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>
                <PEERS>
                    <P ID="1">{i}.spl</P>
                </PEERS>
                <MODELS>
                    <M ID="1">{i}.spl</M>
                </MODELS>
            </EVAL>
        """
        )

    conf_file.write_text(
        f"""
        <ROUGE-EVAL version="1.55">
        {"".join(conf_entries)}
        </ROUGE-EVAL>
    """
    )

    output = subprocess.check_output(
        [f"{rouge_path}/ROUGE-1.5.5.pl", "-e", f"{rouge_path}/data", *args.split(), "-a", str(conf_file)]
    ).decode()

    temp.cleanup()
    return output


def rouge_reference_overall(
    refs: list[str],
    hyps: list[str],
    args: str = "-n 2",
) -> dict[str, float]:
    """

    Run official ROUGE, returning overall ROUGE scores for the entire dataset.
    This will NOT generate individual scores for each model output.

    """

    output = rouge_reference_base(refs, hyps, args)

    pat = r"(ROUGE-.*?) Average_(.): (.*?) \(95\%-conf\.int\. (.*?) - (.*?)\)"
    scores = {}

    for rtype, measure, score, cb, ce in re.findall(pat, output):
        rtype = rtype.lower()
        measure = measure.lower()

        scores[f"{rtype}-{measure}"] = float(score)
        scores[f"{rtype}-{measure}_cb"] = float(cb)
        scores[f"{rtype}-{measure}_ce"] = float(ce)

    return scores


def rouge_data_lead3_baseline(article: str) -> str:
    """

    Run the Lead-3 baseline on an article text.

    """

    import nltk

    return "\n".join(nltk.sent_tokenize(article)[:3])


if __name__ == "__main__":
    split = "validation"

    import datasets
    from tqdm import tqdm

    data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation")

    refs = data["highlights"][:10]
    hyps = [rouge_data_lead3_baseline(s) for s in tqdm(data["article"][:10], desc=f"Generating Hypotheses ({split})")]

    scores = rouge_reference_overall(refs, hyps, "-n 2")

    print(scores)
