
import requests

def pull_ips():

  # Fetch the AWS IP ranges from the JSON file
  url = "https://ip-ranges.amazonaws.com/ip-ranges.json"
  response = requests.get(url)
  data = response.json()
  
  # Extract all `ip_prefix` values
  ip_prefixes = [item['ip_prefix'] for item in data['prefixes'] if 'ip_prefix' in item]
  
  '''
  for ip in ip_prefixes:
    print(f'  IP: {ip}\n')
  '''
  return(ip_prefixes)

def generate_puppet_file(ip_prefixes):
  # Generate the Puppet manifest content
  puppet_content = '''# Puppet manifest generated with AWS IP ranges

  $ip_list = [
  '''

  # Add each IP to the list in Puppet syntax
  for ip in ip_prefixes:
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


def main():
  ip_prefixes = pull_ips()
  generate_puppet_file(ip_prefixes)

if __name__ == "__main__":

  main()
