from fe_modules.parsing import parse
import os

if __name__ == "__main__":
    parse(os.path.join(os.path.abspath(os.getcwd()), '10k_most_freq.csv'),
          os.path.join(os.path.abspath(os.getcwd()), 'sites_out.csv'))
