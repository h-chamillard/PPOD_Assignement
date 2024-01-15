def attributes_identification(df):
    identification = [[0] * 4 for i in range(28)]
    i = 0
    for column in df.columns:
        identification[i][0] = column
        nb_unique_values = df[column].nunique()
        identification[i][1] = nb_unique_values
        nb_total_values = df[column].count()
        identification[i][2] = nb_total_values
        identification[i][3] = (nb_unique_values / nb_total_values)
        i += 1

    identification_ordered = sorted(identification, key=lambda x: x[3], reverse=True)

    # print("Uniqueness Ratio of the unique attributes : ")
    # for i in range(27):
    #     print(identification_ordered[i][0], " : ", identification_ordered[i][3])
    #
    return identification_ordered