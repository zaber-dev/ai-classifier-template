from ai_classifier.core.config import PipelineConfig
from ai_classifier.training.pipeline import TrainingPipeline


def main() -> None:
    config = PipelineConfig.from_yaml("examples/configs/template_config.yaml")
    report = TrainingPipeline(config=config).run()
    print("Training complete")
    print(report)


if __name__ == "__main__":
    main()
