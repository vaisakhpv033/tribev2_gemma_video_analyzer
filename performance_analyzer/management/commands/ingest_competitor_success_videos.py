import os
import glob
import csv
from pathlib import Path
from django.core.management.base import BaseCommand
from django.core.files import File
from performance_analyzer.models import PerformanceVideo
from analyzer.utils.brain_service import analyzer, predictor

class Command(BaseCommand):
    help = "Ingest .npz files for competitor success videos, add impressions, and run brain analysis"

    def handle(self, *args, **options):
        npz_base_folder = Path(r"D:\Work\R_and_D\tribev2\phase2\competitorvideos\output_npz")
        csv_file_path = Path(r"D:\Work\R_and_D\tribev2\phase2\competitorvideos\ADS_Seekers Notes_June's Journey_Triple Match City_2026-06-08_cleaned.csv")
        
        if not npz_base_folder.exists():
            self.stdout.write(self.style.ERROR(f"Directory {npz_base_folder} does not exist."))
            return
            
        if not csv_file_path.exists():
            self.stdout.write(self.style.ERROR(f"CSV file {csv_file_path} does not exist."))
            return

        # 1. Parse CSV and build a dictionary of asset_id -> impressions
        impressions_map = {}
        try:
            with open(csv_file_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asset_id = row.get("asset_id")
                    impressions_str = row.get("impressions")
                    if asset_id and impressions_str:
                        try:
                            impressions_map[asset_id] = int(impressions_str)
                        except ValueError:
                            pass
            self.stdout.write(self.style.SUCCESS(f"Loaded {len(impressions_map)} records from CSV."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error reading CSV: {e}"))
            return

        # 2. Find all .npz files recursively
        npz_files = glob.glob(str(npz_base_folder / "**" / "*.npz"), recursive=True)
        if not npz_files:
            self.stdout.write(self.style.WARNING(f"No .npz files found in {npz_base_folder}."))
            return
            
        self.stdout.write(self.style.NOTICE(f"Found {len(npz_files)} .npz files. Processing..."))

        for npz_file_path in npz_files:
            npz_path_obj = Path(npz_file_path)
            filename = npz_path_obj.name
            
            # Match asset_id from CSV to filename
            matched_asset_id = None
            for asset_id in impressions_map.keys():
                if asset_id in filename:
                    matched_asset_id = asset_id
                    break
                    
            impressions = None
            if matched_asset_id:
                impressions = impressions_map[matched_asset_id]
            else:
                self.stdout.write(self.style.WARNING(f"Could not find impressions for {filename} in CSV."))

            # 3. Get or create the PerformanceVideo record
            video = PerformanceVideo.objects.filter(filename=filename, tier="COMPETITOR_SUCCESS").first()
            created = False
            if not video:
                video = PerformanceVideo(filename=filename, tier="COMPETITOR_SUCCESS")
                # Save the .npz file using Django's File storage to copy it to MEDIA_ROOT
                with open(npz_file_path, 'rb') as f:
                    video.npz_file.save(filename, File(f), save=False)
                created = True
            
            # Update impressions (even if it already existed)
            video.impressions = impressions
            video.save()

            if created:
                self.stdout.write(self.style.SUCCESS(f"Created new record for {filename} with {impressions} impressions."))
            else:
                self.stdout.write(self.style.SUCCESS(f"Record for {filename} already exists, updated impressions. Analyzing again."))

            self.stdout.write(f"Analyzing {filename}...")
            
            try:
                # 4. Extract brain features and timeseries using the original path for analysis
                # (The analyzer might need the original file or the one in media_root. Both should be identical.)
                extraction_result = analyzer.analyze(str(npz_path_obj))
                model_features = extraction_result["model_features"]
                timeseries = extraction_result["timeseries"]

                # 5. Run XGBoost prediction
                prediction = predictor.predict(model_features)

                # 6. Update the database record
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
                
        self.stdout.write(self.style.SUCCESS("Finished ingesting and analyzing competitor success videos."))
