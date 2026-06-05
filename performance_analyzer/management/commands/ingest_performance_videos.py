import os
import glob
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from performance_analyzer.models import PerformanceVideo
from analyzer.utils.brain_service import analyzer, predictor

class Command(BaseCommand):
    help = "Ingest .npz files for top and bottom performing videos and run brain analysis"

    def handle(self, *args, **options):
        media_root = Path(settings.MEDIA_ROOT)
        
        folders_to_process = [
            ("TOP", media_root / "top_5_videos_TTG"),
            ("BOTTOM", media_root / "bottom_5_videos_TTG"),
        ]

        for tier, folder_path in folders_to_process:
            self.stdout.write(self.style.NOTICE(f"Processing {tier} videos from {folder_path}"))
            
            if not folder_path.exists():
                self.stdout.write(self.style.WARNING(f"Directory {folder_path} does not exist. Skipping."))
                continue

            npz_files = glob.glob(str(folder_path / "*.npz"))
            if not npz_files:
                self.stdout.write(self.style.WARNING(f"No .npz files found in {folder_path}."))
                continue

            for npz_file_path in npz_files:
                npz_path_obj = Path(npz_file_path)
                filename = npz_path_obj.name
                
                # Relative path from MEDIA_ROOT for the FileField
                relative_path = npz_path_obj.relative_to(media_root)

                # Get or create the PerformanceVideo record
                video, created = PerformanceVideo.objects.get_or_create(
                    filename=filename,
                    tier=tier,
                    defaults={
                        "npz_file": str(relative_path).replace("\\", "/"),
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f"Created new record for {filename}"))
                else:
                    self.stdout.write(self.style.SUCCESS(f"Record for {filename} already exists, analyzing again."))
                    video.npz_file = str(relative_path).replace("\\", "/")

                self.stdout.write(f"Analyzing {filename}...")
                
                try:
                    # 1. Extract brain features and timeseries
                    extraction_result = analyzer.analyze(str(npz_path_obj))
                    model_features = extraction_result["model_features"]
                    timeseries = extraction_result["timeseries"]

                    # 2. Run XGBoost prediction
                    prediction = predictor.predict(model_features)

                    # 3. Update the database record
                    video.brain_predicted_ctr = prediction.get("predicted_ctr")
                    video.brain_predicted_class = prediction.get("predicted_class")
                    video.brain_predicted_confidence = prediction.get("predicted_confidence")
                    video.brain_prediction_tier = prediction.get("prediction_tier")
                    video.brain_ctr_lower_bound = prediction.get("ctr_lower_bound")
                    video.brain_ctr_upper_bound = prediction.get("ctr_upper_bound")
                    video.brain_model_features = model_features
                    video.brain_timeseries = timeseries
                    
                    video.save()
                    
                    self.stdout.write(self.style.SUCCESS(f"Analysis completed for {filename}: Predicted CTR = {video.brain_predicted_ctr:.2f}%"))

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error analyzing {filename}: {e}"))
                    
        self.stdout.write(self.style.SUCCESS("Finished ingesting and analyzing performance videos."))
