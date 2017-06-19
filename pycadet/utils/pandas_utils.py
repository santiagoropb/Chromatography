import pandas as pd
import numpy as np
import collections
import numbers
import six


def add_row_to_df(df, index, parameters=None):

    if parameters is None:
        if (isinstance(index, list) or isinstance(index, tuple)) and \
                not isinstance(index, six.string_types):
            ndf = pd.DataFrame(np.nan, index=index, columns=df.columns)
            return df.append(ndf)

        else:
            ndf = pd.DataFrame(np.nan, index=[index], columns=df.columns)
            return df.append(ndf)
    else:

        if (isinstance(index, list) or isinstance(index, tuple)) and \
                not isinstance(index, six.string_types) and \
                isinstance(parameters, collections.Sequence):

            assert len(index) == len(parameters)

            ndf = pd.DataFrame(np.nan, index=index, columns=df.columns)
            adf = df.append(ndf)

            for i, cid in enumerate(index):
                params = parameters[i]
                for name, value in params.items():
                    if name not in df.columns:
                        msg = """"{} is not a parameter 
                        of dataframe """.format(name)
                        raise RuntimeError(msg)
                    adf.set_value(cid, name, value)
            return adf
        elif (isinstance(index, six.string_types) or
                  isinstance(index, numbers.Integral)) and \
                isinstance(parameters, dict):

            ndf = pd.DataFrame(np.nan, index=[index], columns=df.columns)
            adf = df.append(ndf)

            for name, value in parameters.items():
                if name not in df.columns:
                    msg = """"{} is not a parameter 
                    of dataframe """.format(name)
                    raise RuntimeError(msg)
                adf.set_value(index, name, value)
            return adf


def set_value_in_df(df, index, name, value):

    if (isinstance(index, list) or isinstance(index, tuple)) and \
            (isinstance(value, list) or isinstance(value, tuple)) and \
            isinstance(name, six.string_types) and \
            not isinstance(index, six.string_types):

        if len(index) != len(value):
            raise RuntimeError("The arrays must be equal size")

        for i, cid in enumerate(index):
            df.set_value(cid, name, value[i])

    elif (isinstance(value, list) or isinstance(value, tuple)) and \
            (isinstance(name, list) or isinstance(name, tuple)) and \
            (isinstance(index, six.string_types) or isinstance(index, numbers.Integral)):

        for i, n in enumerate(name):
            if n not in df.columns:
                msg = """"{} is not a parameter 
                of dataframe """.format(name)
                raise RuntimeError(msg)
            df.set_value(index, n, value[i])

    elif (isinstance(index, six.string_types) or isinstance(index, numbers.Integral)) and \
            isinstance(name, six.string_types) and \
            (isinstance(value, six.string_types) or isinstance(value, numbers.Number)):

        df.set_value(index, name, value)