from fe_modules.parsing import parser
import os

if __name__ == "__main__":
    parser().parse(os.path.join(os.path.abspath(os.getcwd()), 'sites_out_vpn.csv'),
                   os.path.join(os.path.abspath(os.getcwd()), 'sites_out_vpn.xls'),
                   n_partitions=8, timeout=10, max_retries=3, start=400, end=450)
