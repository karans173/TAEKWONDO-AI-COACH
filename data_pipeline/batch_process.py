import os
from data_pipeline.extract import extract_kick_to_csv

def batch_process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    if not video_files:
        print(f"No mp4 files found in {input_dir}.")
        return
        
    print(f"Found {len(video_files)} videos to process.")
    
    success_count = 0
    fail_count = 0
    
    for idx, filename in enumerate(video_files):
        video_path = os.path.join(input_dir, filename)
        csv_filename = filename.replace(".mp4", ".csv")
        output_csv_path = os.path.join(output_dir, csv_filename)

        is_right_leg = False if "sidekick" in filename.lower() else True

        print(f"\n[{idx + 1}/{len(video_files)}] Processing: {filename}")
        
        try:
            extract_kick_to_csv(video_path, output_csv_path, is_right_leg=is_right_leg)
            
            if os.path.exists(output_csv_path):
                success_count += 1
            else:
                print("-> Skipped: No valid kick detected.")
                fail_count += 1
                
        except Exception as e:
            print(f"-> Failed to process {filename}. Error: {e}")
            fail_count += 1

    print(f"\n===============================")
    print(f"Batch Processing Complete!")
    print(f"Successfully Processed: {success_count}")
    print(f"Failed / Skipped: {fail_count}")
    print(f"CSVs saved to: {output_dir}")
    print(f"===============================")

if __name__ == "__main__":
    input_dir = r"data\dataset"
    output_dir = r"data\extracted_csvs"
    batch_process_videos(input_dir, output_dir)