from fe_modules.parsing import parser
import os

if __name__ == "__main__":
    parser().parse(os.path.join(os.path.abspath(os.getcwd()), 'sites_out_turbo_gigachad.csv'),
                   os.path.join(os.path.abspath(os.getcwd()), 'sites_out_turbo_gigachad_new.xls'),
                   n_partitions=8, timeout=10, max_retries=1 )
