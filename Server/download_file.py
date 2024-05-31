import io
import os
import cv2
from google.oauth2 import service_account
from googleapiclient.discovery import build

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import numpy as np

from Server import globalVariables

imageCounts = 0
images = []
labels = []

def createDriveService():
    credentials = service_account.Credentials.from_service_account_file(
        'Server/exalted-pattern-400909-3eaa10f4b2b4.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )

    return build('drive', 'v3', credentials=credentials)

def get_file_id_by_name(file_name, folder_id, drive_service):
    query = f"name='{file_name}' and '{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields='files(id)').execute()
    files = results.get('files', [])

    if not files:
        print(f"No file found with the name '{file_name}' in the folder on drive.", flush=True)
        return None

    return files[0]['id']

def download_model(file_name, folder_id):
    drive_service = createDriveService()

    file_id = get_file_id_by_name(file_name, folder_id, drive_service)

    if file_id:
        request = drive_service.files().get_media(fileId=file_id)
        fh = open(globalVariables.model_file, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}% file {file_name}.", flush=True)
    else:
        return False
            
    return True

def download_model_from_folder(folder_id, save_folder):
    drive_service = createDriveService()

    results = drive_service.files().list(
        q=f"'{folder_id}' in parents",
        fields='files(id, name, mimeType)').execute()

    items = results.get('files', [])

    if not items:
        print('No files found.')
        return False
    else:
        # Create the save folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for item in items:
            file_id = item['id']
            file_name = item['name']
            mime_type = item['mimeType']

            if mime_type == 'application/vnd.google-apps.folder':
                # Recursive call for subfolders
                download_model_from_folder(file_id, os.path.join(save_folder, file_name))
            else:
                # Download file
                request = drive_service.files().get_media(fileId=file_id)
                save_path = os.path.join(save_folder, file_name)
                fh = io.FileIO(save_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Downloaded {file_name} {int(status.progress() * 100)}%")

        return True

def upload_model(file_path, folder_id):
    media = MediaFileUpload(file_path, resumable=True)
    drive_service = createDriveService()

    file_id = get_file_id_by_name(os.path.basename(file_path), folder_id, drive_service)
    if not file_id:
        file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    else:
        drive_service.files().update(fileId=file_id, media_body=media).execute()

def download_file(file_id, file_name, drive_service): #download file from gg drive  
    global imageCounts

    request = drive_service.files().get_media(fileId=file_id)
    fh = open(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% file {file_name}.", flush=True)

    imageCounts += 1


#Get file instead of save to folder
def get_file(file_id, folder_name, drive_service):
    global images
    global labels

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    image_bytes = np.asarray(bytearray(fh.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    images.append(image)
    labels.append(folder_name)

# Get all pages (cause pagination)
def download_all_pages_of_folder(folder_id, folder, drive_service, isSave = False):
    if not os.path.exists(folder) and isSave:
        os.mkdir(folder)

    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        files = response.get('files', [])
        if not files:
            break

        for file in files:
            file_id = file['id']
            file_name = file['name']
            if not isSave:
                get_file(file_id, folder, drive_service)
            else:
                download_file(file_id, f'{folder}/{file_name}', drive_service)

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

def download_folder(folder_id, local_folder_path, isSave = False):
    global imageCounts, images, labels
    imageCounts = 0
    images = []
    labels = []
    
    drive_service = createDriveService()

    if not os.path.exists(local_folder_path) and isSave:
        os.mkdir(local_folder_path)

    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        folders = response.get('files', [])
        if not folders:
            break

        for folder in folders:
            mime_type = folder['mimeType']
            if mime_type == 'application/vnd.google-apps.folder': #only download if it is a subfolder
                folder_id = folder['id']
                folder_name = folder['name']
                if not isSave:
                    download_all_pages_of_folder(folder_id, folder_name, drive_service, isSave)
                else:
                    download_all_pages_of_folder(folder_id, os.path.join(local_folder_path, folder_name), drive_service, isSave)

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break
    
    if not isSave:
        return images, labels
    else:
        return imageCounts
    

    