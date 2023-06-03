import os.path
import re
import sys
import yaml



#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#  References
#   https://www.scrapingbee.com/blog/practical-xpath-for-web-scraping/
#
#------------------------------------------------------------------------------
#
# This script demonstrates how to use YAML anchors and references
#------------------------------------------------------------------------------

file_name = 'conf.yaml'

''' 
server_image_suffix: "live-server-amd64.iso"
desktop_image_suffix: "desktop-amd64.iso"
netboot_image_suffix: "netboot-amd64.tar"
download_image_suffix: ${server_image_suffix}
'''

try:
    file_handle = open(file_name, 'rb')
except OSError:
    print(f' !! Could not open/read file: {fname} !!')
    sys.exit()

with file_handle:
  vars = yaml.safe_load(file_handle)
  # print(vars)
  server_image_suffix = vars['server_image_suffix']
  desktop_image_suffix = vars['desktop_image_suffix']
  netboot_image_suffix = vars['netboot_image_suffix']
  download_image_suffix = vars['download_image_suffix']
  
print(f'server_image_suffix: {server_image_suffix}')
print(f'desktop_image_suffix: {desktop_image_suffix}')
print(f'netboot_image_suffix: {netboot_image_suffix}')
print(f'download_image_suffix: {download_image_suffix}')
  
 
