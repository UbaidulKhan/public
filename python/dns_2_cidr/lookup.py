import socket
from ipwhois import IPWhois



def resolve_ipv4_addresses(domain_name):
    try:
        # Get all addresses (IPv4 and IPv6)
        addr_info = socket.getaddrinfo(domain_name, None)
        
        # Filter only IPv4 addresses (family = socket.AF_INET)
        ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
        
        return ipv4_addresses
    except socket.gaierror as e:
        print(f"Error resolving domain {domain_name}: {e}")
        return None



# Step 1: Resolve the domain name to an IP address
def resolve_domain_to_ip(domain_name):
    try:
        ip_address = socket.gethostbyname(domain_name)
        return ip_address
    except socket.gaierror as e:
        print(f"Error resolving domain {domain_name}: {e}")
        return None

# Step 2: Perform WHOIS lookup on the IP address
def whois_lookup(ip_address):
    try:
        obj = IPWhois(ip_address)
        result = obj.lookup_rdap()  # you can also use obj.lookup_whois() for older WHOIS
        return result
    except Exception as e:
        print(f"Error performing WHOIS lookup for IP {ip_address}: {e}")
        return None


domains = [ 'login.salesforce.com', 'gmuadvancement.my.salesforce.com',
            ' gmuadvancement--gmusandbox.sandbox.my.salesforce.com',
            'mason.my.salesforce.com', 'gmuadvancement.my.salesforce.com',
            'test.l2.salesforce.com', 'na168-ia5.salesforce.com',
            'st1.edge.sfdc-yfeipo.edge2.salesforce.com',
            'www.salesforce.com', 'gmuadvancement.my.salesforce.com',
            'salesforce.com' ]

for domain in domains:  
  print(f"-----------------------------------------------------------")
  print(f'  Looking up domain: {domain}')
  print(f"-----------------------------------------------------------")
  
  ipv4_list = resolve_ipv4_addresses(domain)
  if ipv4_list:
    # print(f"IPv4 Addresses for {domain}: {ipv4_list}")
    for ip in ipv4_list:
      print(f'  Looking up whois for IP: {ip}')
      whois_info = whois_lookup(ip)
      if whois_info:
        # print(whois_info.get('asn_cidr', {}))
        print(f"    ", domain, ip, "CIDR:", whois_info.get('asn_cidr', {}), "\n")

  print(f"\n\n")



