from pathlib import Path

from src.summarize_unity_results import summarize_evaluation


def test_summarize_evaluation_creates_report(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "eval"
    evaluation_root.mkdir()
    (evaluation_root / "final_summary.json").write_text(
        '{"summary":{"sequence_count":1,"successful_sequences":1,"failed_sequences":0,"average_candidate_recall":1.0,"average_accepted_detection_recall":0.1,"average_filtered_mae_px":4.5,"average_locked_frame_percentage":25.0,"average_effective_processing_fps":8.0}}',
        encoding="utf-8",
    )
    (evaluation_root / "final_metrics.csv").write_text(
        "sequence_dir,candidate_recall,accepted_detection_recall,filtered_mae_px,locked_frame_percentage,motion_quality_note\n"
        "outputs/unity-evaluation/run/sequence_008,1.0,0.1,4.5,25.0,review reacquisition\n",
        encoding="utf-8",
    )

    output_path = summarize_evaluation(evaluation_root, tmp_path / "report.md", "Test Report")

    report = output_path.read_text(encoding="utf-8")
    assert "# Test Report" in report
    assert "sequence_008" in report
    assert "Accepted detection recall: 0.100000" in report
    assert "review reacquisition" in report
