from fe_modules.parsing import parser
import os

if __name__ == "__main__":
    parser().parse('./to_check.csv',
                   os.path.join(os.path.abspath(os.getcwd()), './sites_new.xls'),
                   n_partitions=8, timeout=10, max_retries=1)
