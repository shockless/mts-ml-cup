from fe_modules.parsing import parse
import os

if __name__ == "__main__":
    parse(os.path.join(os.path.abspath(os.getcwd()), 'top_10k_urls.csv'),
          os.path.join(os.path.abspath(os.getcwd()), 'sites_out.xls'))
