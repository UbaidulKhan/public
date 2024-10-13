import ipaddress

import ipaddress

def get_network_subnet_from_ip(ip_address):
  """Gets the network and subnet mask from an IP address.

  Args:
    ip_address: The IP address as a string.

  Returns:
    A tuple containing the network address and subnet mask as strings, or None if invalid.
  """

  try:
    ip_obj = ipaddress.ip_address(ip_address)
    network = ip_obj.network  # This line is incorrect
  except ValueError:
    print("Invalid IP address:", ip_address)
    return None, None

  # We need to convert the IP address to a network object first
  network = ipaddress.Network(ip_obj)
  subnet_mask = network.netmask

  return network.with_prefixlen, subnet_mask.with_prefixlen

def read_ips_from_file(filename):
  """Reads IP addresses from a file and returns a list.

  Args:
    filename: The name of the file containing the IP addresses.

  Returns:
    A list of IP addresses.
  """

  ips = []
  with open(filename, "r") as f:
    for line in f:
      ip = line.strip()
      ips.append(ip)

  return ips

# Example usage:
filename = "/home/ukhan/Development/github/public.git/python/cidr_maker/sf_ips_cln.txt"
ip_addresses = read_ips_from_file(filename)

for ip in ip_addresses:
  network, subnet_mask = get_network_subnet_from_ip(ip)
  if network and subnet_mask:
    print(f"IP {ip} belongs to network {network} with subnet mask {subnet_mask}")
  else:
    print(f"Invalid IP address: {ip}")
