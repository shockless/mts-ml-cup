from fe_modules.parsing import parser
import os

if __name__ == "__main__":
    parser().parse(os.path.join(os.path.abspath(os.getcwd()), 'sites_out_vpn_only.csv'),
          os.path.join(os.path.abspath(os.getcwd()), 'sites_out_2.xls'),
          5)
