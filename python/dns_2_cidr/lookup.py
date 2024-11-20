import socket
import ipaddress
from ipwhois import IPWhois

#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------ 


#
## This sub-routine will add header and trailer to each line from
## the ip_collection
##-----------------------------------------------------------------------------
#
# 
def sort_ip_addresses(ip_list):
    """
    Sorts a list of IP addresses.

    Args:
        ip_list (list of str): List of IP addresses as strings.

    Returns:
        list of str: The sorted list of IP addresses.
    """
    
    sorted_list = sorted(ip_list, key=lambda ip: ipaddress.ip_network(ip, strict=False))
    # print(f'  Sorted list is: {sorted_list}')
    
    return(sorted_list)


#
## This sub-routine will add header and trailer to each line from
## the ip_collection
##-----------------------------------------------------------------------------
#
# 
def print_collection(ip_collection):
  
  collection_hash = hash(tuple(ip_collection))
  
  print(f' |  Hash of the collectino: {collection_hash}')
  print(f' +-------------------------------------------------')

  header='  - '
  for ip in ip_collection:
    print(header + "'" + ip + "'")
#
## This sub-routine will insert the network address into 
## a list, if it does not already exist
##-----------------------------------------------------------------------------
#
# 
def insert_if_not_exists(ip, ip_collection):
    """
    Inserts an item into a list or queue if it does not already exist.

    Args:
        item: The item to be added.
        collection: The list or queue to check and insert into.

    Returns:
        bool: True if the item was added, False otherwise.
    """
    print(f' {ip} Type of ips is: {type(ip)}')
    
    if ip not in ip_collection:
        #print(f' {ip} does not exist in ip_collection, adding')
        ip_collection.append(ip)
    #    return True
    #return False
    else:
        print(f' {ip} alredy exist in ip_collection, skipping')

    

    
#
## This sub-routine will collect all the IPv4 addresses, returned by 
## name service lookup.
##-----------------------------------------------------------------------------
#
# 
def resolve_ipv4_addresses(domain_name):
  
  try:
    # Get all addresses (IPv4 and IPv6)
    addr_info = socket.getaddrinfo(domain_name, None)
        
    # Filter only IPv4 addresses (family = socket.AF_INET)
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    
    # print(f' IP addres: {ipv4_addresses} is of type: {type(ipv4_addresses)}')
    
    return (ipv4_addresses)
  
  except socket.gaierror as e:
    print(f"Error resolving domain {domain_name}: {e}")
    return None



#
## This sub-routine take one IP address and perform a ASN look-up. Then
## the retrieved information is returned. 
##-----------------------------------------------------------------------------
#
# 
# Step 2: Perform WHOIS lookup on the IP address
def whois_lookup(ip_address):
  try:
    obj = IPWhois(ip_address)
    result = obj.lookup_rdap()  # you can also use obj.lookup_whois() for older WHOIS
    return result
  except Exception as e:
    print(f"Error performing WHOIS lookup for IP {ip_address}: {e}")
    return None


#
## main()
##-----------------------------------------------------------------------------
#
#
def main():

  
  #
  ## Build a list to store network addresses
  ip_collection = []
    
  #
  ## Build a list of domain names to lookup
  domains = [ 'login.salesforce.com', 'gmuadvancement.my.salesforce.com',
              ' gmuadvancement--gmusandbox.sandbox.my.salesforce.com',
              'mason.my.salesforce.com', 'gmuadvancement.my.salesforce.com',
              'test.l2.salesforce.com', 'na168-ia5.salesforce.com',
              'st1.edge.sfdc-yfeipo.edge2.salesforce.com',
              'www.salesforce.com', 'gmuadvancement.my.salesforce.com',
              'salesforce.com' ]
              
  #
  ## Iterate over the list:
  # 1) Resolve the domain to IP address - get all the IP addresses.
  # 2) For each IP address, perform a ASN lookup, by calling
  #     whois_lookup
  #
  for domain in domains:  
    print(f"-----------------------------------------------------------")
    print(f'  Looking up domain: {domain}')
    print(f"-----------------------------------------------------------")
  
    ipv4_list = resolve_ipv4_addresses(domain)
    if ipv4_list:
      # print(f"IPv4 Addresses for {domain}: {ipv4_list}")
      for ip in ipv4_list:
        # print(f'  Looking up whois for IP: {ip}')
        whois_info = whois_lookup(ip)
        if whois_info:
          # print(whois_info.get('asn_cidr', {}))
          # print(f"    ", domain, ip, "CIDR:", whois_info.get('asn_cidr', {}), "\n")
          ip = whois_info.get('asn_cidr', {})
          insert_if_not_exists(ip, ip_collection)

      print(f"\n\n")
  
  #
  ## Print the list
  # print(ip_collection)
  ip_collection = sort_ip_addresses(ip_collection) 
  print_collection(ip_collection)



if __name__ == "__main__":
  main()  

