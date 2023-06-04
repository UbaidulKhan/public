import requests # request img from web
import shutil # save img locally
import hashlib

#
#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#  References
#    https://pynative.com/python-regex-capturing-groups/   
#
#------------------------------------------------------------------------------
#
# This script demonstrates downloading files & calculating sha256 checksum
#------------------------------------------------------------------------------



ubuntu_url = 'https://releases.ubuntu.com/23.04/ubuntu-23.04-netboot-amd64.tar.gz'
# url = input('Please enter an image URL (string):') #prompt user for img url
url = ubuntu_url
# file_name = input('Save image as (string):') #prompt user for file_name
file_name = 'ubuntu-23.04-netboot-amd64.tar.gz'
sha256_checksum = ''
res = requests.get(url, stream = True)

if (res.status_code == 200):
  try:
    with open(file_name,'wb') as f:
      shutil.copyfileobj(res.raw, f)

  except Exception as e:
    print(f' There was an error writing: {file_name}')
    print(f' Please ensure you have write permission on the directory & there is enough space on the disk')
    print(f'  {e}')
    sys.exit(1)
    
  finally:
    print(f' Successfully downloaded: {file_name}')
    
    try:
      # Open,close, read file and calculate MD5 on its contents 
      with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        sha256_checksum = hashlib.sha256(data).hexdigest()
          
    except Exception as e:
      print(f' Failed to calculate {file_name} checksum')
      print(f' Please ensure file exists and readable')
      print(f'  {e}')
      sys.exit(1)
      
    finally:
      print(f' File {file_name} checksum: \n   {sha256_checksum}')
    
      
else:
  print('Image Couldn\'t be retrieved')
