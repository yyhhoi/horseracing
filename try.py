
from lib.datagenerator import DataGenerator



dfpth = "data/training_dataframes/processed_df.pickle"
dg = DataGenerator(df_path=dfpth)


for train_info, test_info in dg.iterator():
    (x_train, xseq_train, y_train, racecode_train) = train_info
    (x_test, xseq_test, y_test, racecode_test) = test_info
    break