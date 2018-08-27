import sys
import datetime
from openpyxl import load_workbook


TARGET_CATEGORY = 'Ease of Doing Business Ranking'
TITLES = ['Geography', 'Category', 'Data', 'Type', 'Unit', 'Current', 'Constant', '2012', '2013', '2014', '2015',
          '2016', '2017', '2018', '2019', '2020']


class TrainData:
    def __init__(self, debug=False):
        now = datetime.datetime.now()
        self.cur_year = now.year
        self.debug = debug

    @staticmethod
    def read_xlsx(xlsx_path, sheet_idx=0):
        workbook = load_workbook(xlsx_path, data_only=True)
        sheet_name = workbook.get_sheet_names()[sheet_idx]
        worksheet = workbook.get_sheet_by_name(sheet_name)

        sys.stdout.write("first_sheet: {}\n".format(sheet_name))

        raw_data = []
        for row in worksheet.iter_rows():
            line = []
            for cell in row:
                line.append(cell.value)
            raw_data.append(line)

        return raw_data, sheet_name

    def cleaning(self, raw_data):
        # check the first line titles ---------------------------------------------------------------------
        value_cols_idxs = []
        category_col_idx = -1
        first_raw = raw_data[0]
        for title in first_raw:
            try:
                if title.isdigit() and self.cur_year > int(title) > 2000:
                    value_cols_idxs.append(first_raw.index(title))

                if title.lower() == 'Category'.lower():
                    category_col_idx = (first_raw.index(title))
            except Exception as e:
                print(e)
                break

        target_row_idx = -1
        for i in range(1, len(raw_data)):
            line = raw_data[i]
            if line[category_col_idx].replace(' ', '').lower() == TARGET_CATEGORY.replace(' ', '').lower():
                target_row_idx = i
                break

        if self.debug:
            sys.stdout.write("category_col_idx: {}\n".format(category_col_idx))
            sys.stdout.write("target_row_idx: {}\n".format(target_row_idx))

        # crop the value area -----------------------------------------------------------------
        x_data = []
        y_data = []
        for i in range(1, len(raw_data)):
            try:
                if i == target_row_idx:
                    for j in value_cols_idxs:
                        y_data.append(raw_data[i][j])
                    continue

                value_line = []
                for j in value_cols_idxs:
                    value_line.append(raw_data[i][j])

                x_data.append(value_line)
            except Exception as e:
                print(e)

        # swap rows and cols on x_data ----------------------------------------------------------------------------
        swap_x_data = []
        for j in range(len(x_data[0])):
            new_line = []
            for i in range(len(x_data)):
                new_line.append(x_data[i][j])
            swap_x_data.append(new_line)

        titles = [raw_data[0][j] for j in value_cols_idxs]
        categories = [raw_data[i][category_col_idx] for i in range(1, len(raw_data))]

        return swap_x_data, y_data, titles, categories


if __name__ == '__main__':
    _path = "../data/boi_analysis(1).xlsx"
    td = TrainData()
    _raw_data, _sheet_name = td.read_xlsx(xlsx_path=_path)
    _x_data, _y_data, _titles, _categories = td.cleaning(raw_data=_raw_data)
    print(_titles)
    print(_categories)
    print(_y_data)
    for _line in _x_data:
        print(_line)
