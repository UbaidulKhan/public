
import requests

class AwsIPS:

  def __init__(self, url):
  
    # URL to fetch ips from
    self.url = url
    self.ip_prefixes = []
      
  def get_ips(self):
  
    # Fetch the AWS IP ranges from the JSON file
    # url = "https://ip-ranges.amazonaws.com/ip-ranges.json"
    
    print(f' URL to call: {self.url}')
    response = requests.get(self.url)
    data = response.json()
    
    # Extract all `ip_prefix` values
    self.ip_prefixes = [item['ip_prefix'] for item in data['prefixes'] if 'ip_prefix' in item]
    

  
  # def generate_puppet_file(ip_prefixes):
  def generate_puppet_file(self):

    # Generate the Puppet manifest content
    puppet_content = '''# Puppet manifest generated with AWS IP ranges
  
    $ip_list = [
    '''
  
    # Add each IP to the list in Puppet syntax
    for ip in self.ip_prefixes:
        puppet_content += f"  '{ip}',\n"
  
    # Close the array
    puppet_content += '''\n]
  
    # Example usage of the IP list in a Puppet file resource
    file { '/tmp/aws_ip_list.txt':
      ensure  => 'present',
      content => inline_template('<%= @ip_list.join("\\n") %>'),
    }
    '''
  
    # Write the content to a .pp file (Puppet manifest file)
    with open("aws_ip_list.pp", "w") as file:
        file.write(puppet_content)
  
    print("Puppet manifest file 'aws_ip_list.pp' generated successfully!")
  
#
##   
def main():
  
  url = "https://ip-ranges.amazonaws.com/ip-ranges.json"
  
  myaws = AwsIPS(url)
  myaws.get_ips()
  myaws.generate_puppet_file()
  

  
#
##  
if __name__ == "__main__":

  main()
