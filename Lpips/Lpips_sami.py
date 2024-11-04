import os
import lpips
import torch
from PIL import Image
from torchvision import transforms
import itertools
import pandas as pd
from openpyxl import Workbook, load_workbook

#compute LPIPS scores between images in a folder
def Lpips_compute(dataset_path, results_file):
    loss_fn = lpips.LPIPS(net='alex')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    results = []
    for img1_path, img2_path in itertools.combinations(image_files, 2):
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)

        with torch.no_grad():
            distance = loss_fn(img1, img2)

        results.append([os.path.basename(img1_path), os.path.basename(img2_path), distance.item()])
        print(f"Computed LPIPS between {img1_path} and {img2_path}: {distance.item()}")

    df = pd.DataFrame(results, columns=["Image1", "Image2", "LPIPS_Distance"])
    return df

def save_to_excel(df, output_file, sheet_name):
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_file, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Main function to calculate LPIPS scores for all models in Images folder
def main(images_folder, output_file, folder_to_calculate=None):
    # If a specific folder is given, only calculate for that folder
    if folder_to_calculate:
        dataset_path = os.path.join(images_folder, folder_to_calculate)
        df = Lpips_compute(dataset_path, output_file)
        save_to_excel(df, output_file, sheet_name=folder_to_calculate)
    else:
        # Loop over each folder in the Images directory
        for model_folder in os.listdir(images_folder):
            model_path = os.path.join(images_folder, model_folder)
            if os.path.isdir(model_path):
                df = Lpips_compute(model_path, output_file)
                save_to_excel(df, output_file, sheet_name=model_folder)

    # Compute the average LPIPS scores per model and save to a separate sheet
    workbook = load_workbook(output_file)
    average_lpips_data = []
    for sheet in workbook.sheetnames:
        df = pd.read_excel(output_file, sheet_name=sheet)
        avg_lpips_score = df["LPIPS_Distance"].mean()
        average_lpips_data.append([sheet, avg_lpips_score])

    avg_df = pd.DataFrame(average_lpips_data, columns=["Model", "Average_LPIPS_Score"])
    save_to_excel(avg_df, output_file, sheet_name='Average_Scores')

if __name__ == "__main__":
    images_folder = os.path.expanduser("/home/sami-zagha/Desktop/Lpips/Images")
    output_file = os.path.expanduser("/home/sami-zagha/Desktop/Lpips/Output/lpips_results.xlsx")
    folder_to_calculate = None  # Set to specific folder name if needed
    main(images_folder, output_file, folder_to_calculate)
