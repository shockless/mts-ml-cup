from fe_modules.parsing import parse
import os

if __name__ == "__main__":
    parse(os.path.join(os.path.dirname(__file__), '\\all_urls.csv'),
          os.path.join(os.path.dirname(__file__), '\\sites_out.csv'))
