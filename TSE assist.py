import os
import sys
import glob
import tqdm
import logging
import numpy as np
import pandas as pd
import pytse_client as tse

pd.set_option('future.no_silent_downcasting', True)

logger = logging.Logger("PYTSE")

class DataLoader:
    def __init__(self, symbols: list, fileName: str='output', typeData: str='client', periods: list = None, **kwargs) -> None:
        if isinstance(symbols, str) or isinstance(symbols, list):
            if symbols == 'all':
                symbols = list(tse.symbols_data.all_symbols())
            if isinstance(symbols, str) and symbols in list(tse.symbols_data.all_symbols()):
                symbols = [symbols]
        else:
            raise ValueError('symbols are not valid')
        self.symbols = symbols
        if not isinstance(fileName, str):
            raise ValueError(f'fileName must be a string not a {type(fileName)}')
        if '.xlsx' not in fileName:
            fileName += '.xlsx'
        self.fileName = fileName
        try:
            if kwargs.get('MKDIR', True):
                os.mkdir(os.path.join('.', fileName[:-5]))
        except:
            print(f"Cannot create a file when that file already exists: '{fileName[:-5]}'")
        if not isinstance(typeData, str):
            raise ValueError(f'typeData must be a string not a {type(typeData)}')
        if typeData not in ['client', 'history', 'client&history']:
            raise ValueError(f'typeData must be one of these ["client", "histoty", "client & history"] not "{typeData}"')
        if periods is not None:
            if not isinstance(periods, list):
                raise ValueError(f'periods must be a list not a {type(periods)}')
        else:
            periods = [[0, 1]]
        
        self.periods = periods
        self.typeData = typeData
        self.unavailableSymbolsCTD = []
        self.availableSymbolsCTD = []
        self.unavailableSymbolsHTD = []
        self.availableSymbolsHTD = []

    def __call__(self):
        if self.typeData == 'client':
            self.dataScreamingCTD()
            CTD = self.clientTypeData()
            all_df = []
            for period in CTD:
                df = CTD[period]
                df = pd.DataFrame(
                    df,
                    columns=[
                        'symbol',
                        'vol', 
                        'sum_corporative_buy_volume', 
                        'sum_corporative_sell_volume', 
                        'sum_individual_sell_count', 
                        'sum_individual_buy_count',
                        'date',
                    ]
                )
                all_df.append(df)
                df.to_excel(os.path.join('.', self.fileName[:-5], f'{period}_CTD_{self.fileName}'), index=False, header=True)
            if len(all_df) == 2:
                relativeDF_CTD = self.relativeEtractDataCTD(all_df[0], all_df[1])
                relativeDF_CTD.to_excel(os.path.join('.', self.fileName[:-5], f'{self.periods[0]}_{self.periods[1]}_{self.fileName}'), index=False, header=True)
        elif self.typeData == 'history':
            self.dataScreamingHTD()
            HTD = self.historyTypeData()
            all_df = []
            for period in HTD:
                df = HTD[period]
                df = pd.DataFrame(
                    df,
                    columns=[
                        'date',
                        'open',
                        'close',
                        'max_close_or_open',
                        'min_close_or_open',
                        'open_real',
                        'close_real',
                    ]
                )
                all_df.append(df)
                df.to_excel(os.path.join('.', self.fileName[:-5], f'{period}_HTD_{self.fileName}'), index=False, header=True)
        elif self.typeData == 'client&history':
            self.dataScreamingCTD()
            self.dataScreamingHTD()
            self.symbols = list(set(self.availableSymbolsCTD).intersection(self.availableSymbolsHTD))
            CTD = self.clientTypeData()
            HTD = self.historyTypeData()
            all_df_CTD = []
            all_df_HTD = []
            for periodCTD, periodHTD in zip(CTD, HTD):
                dfCTD = CTD[periodCTD]
                dfHTD = HTD[periodHTD]
                dfCTD = pd.DataFrame(
                    dfCTD,
                    columns=[
                        'symbol',
                        'vol', 
                        'sum_corporative_buy_volume', 
                        'sum_corporative_sell_volume', 
                        'sum_individual_sell_count', 
                        'sum_individual_buy_count',
                        'date client',
                    ]
                )
                all_df_CTD.append(dfCTD)
                dfHTD = pd.DataFrame(
                    dfHTD,
                    columns=[
                        'date history',
                        'open',
                        'close',
                        'max_close_or_open',
                        'min_close_or_open',
                        'open_real',
                        'close_real',
                    ]
                )
                all_df_HTD.append(dfHTD)
                df_total = pd.concat([dfCTD, dfHTD], axis=1)
                df_total.to_excel(os.path.join('.', self.fileName[:-5], f'{periodCTD}_TOTAL_{self.fileName}'), index=False, header=True)
            if len(all_df_CTD) == 2:
                relativeDF_CTD = self.relativeEtractDataCTD(all_df_CTD[0], all_df_CTD[1])
                relativeDF_HTD = self.relativeEtractDataHTD(all_df_HTD[0], all_df_HTD[1])
                relativeDF_CTD_HTD = pd.concat([relativeDF_CTD, relativeDF_HTD], axis=1)
                relativeDF_CTD_HTD.to_excel(os.path.join('.', self.fileName[:-5], f'{self.periods[0]}_{self.periods[1]}_{self.fileName}'), index=False, header=True)
        else:
            raise ValueError(f'typeData must be one of these ["client", "histoty", "client & history"] not "{self.typeData}"')
    
    def disablePrint(self) -> None:
        sys.stdout = open(os.devnull, 'w')
    
    def enablePrint(self) -> None:
        sys.stdout = sys.__stdout__
    
    def clientTypeData(self) -> dict:
        outputs = {}
        for i, _ in enumerate(self.periods):
            outputs[f'period_{i}'] = []
        for symbol in self.symbols:
            for i, period in enumerate(self.periods):
                df = pd.read_csv(os.path.join('.', 'client_types_data', f'{symbol}.csv'))
                outputs[f'period_{i}'].append(self.extractDataCTD(symbol, df, period[0], period[1]))
        return outputs

    def downloadCTD(self) -> None:
        for symbol in tqdm.tqdm(self.symbols):
            if not os.path.exists(os.path.join('.', 'client_types_data', f'{symbol}.csv')):
                try:
                    self.disablePrint()
                    _ = tse.download_client_types_records(symbol, write_to_csv=True)
                    self.enablePrint()
                except Exception as e:
                    logger.warning(f'An Unexpected Error Occurd: {e}')
    
    def extractDataCTD(self, symbol: str, df: pd.DataFrame, startDay: int, numDays: int):
        date = df.at[0, 'date']
        df = self.preprocessingCTD(df)
        vol = np.array(0, dtype=np.float64)
        sum_corporative_buy_volume = np.array(0, dtype=np.float64)
        sum_corporative_sell_volume = np.array(0, dtype=np.float64)
        sum_individual_sell_count = np.array(0, dtype=np.float64)
        sum_individual_buy_count = np.array(0, dtype=np.float64)
        for i in range(startDay, startDay + numDays):
            buy_vol = np.array(df.at[i, 'individual_buy_vol'], dtype=np.float64)
            cor_buy_vol = np.array(df.at[i, 'corporate_buy_vol'], dtype=np.float64)
            vol += np.add(buy_vol, cor_buy_vol)
            sum_corporative_buy_volume += cor_buy_vol
            sum_corporative_sell_volume += np.array(df.at[i, 'corporate_sell_vol'], dtype=np.float64)
            sum_individual_sell_count += np.array(df.at[i, 'individual_sell_count'], dtype=np.float64)
            sum_individual_buy_count += np.array(df.at[i, 'individual_buy_count'], dtype=np.float64)
        
        return [
            symbol,
            vol, 
            sum_corporative_buy_volume, 
            sum_corporative_sell_volume, 
            sum_individual_sell_count, 
            sum_individual_buy_count,
            date,
        ]

    def dataScreamingCTD(self) -> None:
        pathCTD = os.path.join('.', 'client_types_data', '*')
        for file in glob.glob(pathCTD):
            check = True
            df = pd.read_csv(file)
            symbol = file.split(os.sep)[-1]
            symbol = symbol.split('.')[0]
            if len(df.index) < (self.periods[0][0] + self.periods[0][1]):
                check = False
            if len(self.periods) == 2:
                if len(df.index) < (self.periods[1][0] + self.periods[1][1]):
                    check = False
            if check:
                self.availableSymbolsCTD.append(symbol)
            else:
                self.unavailableSymbolsCTD.append(symbol)
        for i, symbol in enumerate(self.unavailableSymbolsCTD):
            print(f'{i}. {symbol}')
        print(f'\n%{(len(self.unavailableSymbolsCTD)/(len(self.availableSymbolsCTD) + len(self.unavailableSymbolsCTD)))*100:.4f} of data was not available.')
        self.symbols = self.availableSymbolsCTD

    def relativeEtractDataCTD(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
        relativeDF = pd.DataFrame(
            columns=[
                'symbol' , 
                'vol' , 
                'vol2' , 
                'vol ratio' , 
                'bv', 
                'sib', 
                'sv',
                'sis', 
                'bv2', 
                'sib2',
                'sv2',
                'sis2',
                'date',
            ]
        )

        relativeDF['symbol'] = df_1['symbol']

        
        relativeDF['vol'] = df_1['vol']
        relativeDF['vol2'] = df_2['vol']
        relativeDF['vol ratio'] = df_1['vol'] / df_2['vol']
        relativeDF['vol ratio'] = relativeDF['vol ratio'].fillna(0)
        
        
        relativeDF['bv'] = df_1['sum_corporative_buy_volume']
        relativeDF['sib'] = df_1['sum_individual_buy_count']
        relativeDF['sv'] = df_1['sum_corporative_sell_volume']
        relativeDF['sis'] = df_1['sum_individual_sell_count']
        
        relativeDF['bv2'] = df_2['sum_corporative_buy_volume']
        relativeDF['sib2'] = df_2['sum_individual_buy_count']
        relativeDF['sv2'] = df_2['sum_corporative_sell_volume']
        relativeDF['sis2'] = df_2['sum_individual_sell_count']

        relativeDF['date'] = df_1['date client']

        return relativeDF

    def preprocessingCTD(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            try:
                df[col] = df[col].astype('float64')
            except:
                if col != 'date':
                    print(f'Could not preprocess data of "{col}" column')
        return df

    def historyTypeData(self):
        outputs = {}
        for i, _ in enumerate(self.periods):
            outputs[f'period_{i}'] = []
        for symbol in self.symbols:
            for i, period in enumerate(self.periods):
                df = pd.read_csv(os.path.join('.', 'tickers_data', f'{symbol}.csv'))
                outputs[f'period_{i}'].append(self.extractDataHTD(symbol, df, period[0], period[1]))
        return outputs
    
    def extractDataHTD(self, symbol: str, df: pd.DataFrame, startDay: int, numDays: int):
        df.sort_values(by=['date'], inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)
        close = np.array(df.at[startDay, 'adjClose'], dtype=np.float64)
        open = np.array(df.at[startDay + numDays, 'adjClose'], dtype=np.float64)
        open_real = np.array(df.at[startDay + numDays - 1, 'open'], dtype=np.float64)
        close_real = np.array(df.at[startDay, 'close'], dtype=np.float64)
        max_close_or_open = np.max(
                [
                    df['open'][startDay:startDay+numDays].to_numpy().astype(np.float32).max(),
                    df['close'][startDay:startDay+numDays].to_numpy().astype(np.float32).max(),
                ]
        )
        min_close_or_open = np.min(
                [
                    df['open'][startDay:startDay+numDays].to_numpy().astype(np.float32).min(),
                    df['close'][startDay:startDay+numDays].to_numpy().astype(np.float32).min(),
                ]
        )
        temp_df = df[startDay:startDay+numDays]
        temp_df.reset_index(drop=True, inplace=True)
        date = df.at[0, 'date']
        return [
            date,
            open,
            close,
            max_close_or_open,
            min_close_or_open,
            open_real,
            close_real,
        ]
    
    def downloadHTD(self) -> None:
        for symbol in tqdm.tqdm(self.symbols):
            if not os.path.exists(os.path.join('.', 'tickers_data', f'{symbol}.csv')):
                try:
                    self.disablePrint()
                    _ = tse.download(symbol, write_to_csv=True)
                    self.enablePrint()
                except Exception as e:
                    logger.warning(f'An Unexpected Error Occurd: {e}')
    
    def dataScreamingHTD(self) -> None:
        pathHTD = os.path.join('.', 'tickers_data', '*')
        for file in glob.glob(pathHTD):
            check = True
            df = pd.read_csv(file)
            symbol = file.split(os.sep)[-1]
            symbol = symbol.split('.')[0]
            if len(df.index) < (self.periods[0][0] + self.periods[0][1] + 2):
                check = False
            if len(self.periods) == 2:
                if len(df.index) < (self.periods[1][0] + self.periods[1][1] + 2):
                    check = False
            if check:
                self.availableSymbolsHTD.append(symbol)
            else:
                self.unavailableSymbolsHTD.append(symbol)
        for i, symbol in enumerate(self.unavailableSymbolsHTD):
            print(f'{i}. {symbol}')
        print(f'\n%{(len(self.unavailableSymbolsHTD)/len(self.symbols))*100:.4f} of data was not available.')
        self.symbols = self.availableSymbolsHTD
    
    def relativeEtractDataHTD(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
        relativeDF = pd.DataFrame(
            columns=[
                'date',
                'open',
                'close',
                'open2',
                'close2',
                'open_real',
                'close_real',
                'open_real2',
                'close_real2',
                'oc_ratio',
                'oc_ratio2',
                'HI',
                'HI2',
                'max_close_or_open_1',
                'max_close_or_open_2',
                'min_close_or_open_1',
                'min_close_or_open_2',
            ]
        )

        relativeDF['date'] = df_1['date history']

        relativeDF['open'] = df_1['open']
        relativeDF['open2'] = df_2['open']

        relativeDF['close'] = df_1['close']
        relativeDF['close2'] = df_2['close']

        relativeDF['open_real'] = df_1['open_real']
        relativeDF['open_real2'] = df_2['open_real']

        relativeDF['close_real'] = df_1['close_real']
        relativeDF['close_real2'] = df_2['close_real']

        relativeDF['oc_ratio'] = df_1['open_real'] / df_1['close_real']
        relativeDF['oc_ratio2'] = df_2['open_real'] / df_2['close_real']

        relativeDF['HI'] = df_1['max_close_or_open'] / df_1['min_close_or_open']
        relativeDF['HI2'] = df_2['max_close_or_open'] / df_2['min_close_or_open']

        relativeDF['max_close_or_open_1'] = df_1['max_close_or_open']
        relativeDF['max_close_or_open_2'] = df_2['max_close_or_open']

        relativeDF['min_close_or_open_1'] = df_1['min_close_or_open']
        relativeDF['min_close_or_open_2'] = df_2['min_close_or_open']


        return relativeDF

    def wrapping(self, periodBlocks, outputNames):
        base_df = None
        __ffilename = ''
        for periodBlock, fileName in zip(periodBlocks, outputNames):
            __ffilename += f'_{fileName}_'
            __filename = os.path.join('.', fileName, f'{periodBlock[0]}_{periodBlock[1]}_{fileName}.xlsx')
            df = pd.read_excel(__filename)
            if base_df is None:
                base_df = df
            else:
                base_df = pd.concat(
                    (
                        base_df,
                        df,
                    ),
                    axis=0,
                )
        base_df.to_excel(os.path.join('.', f'wrapping{__ffilename}{self.fileName[-5:]}'), index=False, header=True)

if __name__ == '__main__':
    while True:
        print('Options:\n')
        print('1- download\n')
        print('2- client & history\n')
        print('3- exit\n')

        _input = int(input('> '))
        if _input == 1:
            typeData = 'client&history'
            periodBlocks = []
            outputNames = []
            nBlocks = 1
            for i in range(nBlocks):
                firstDay_1 = 1
                numDays_1 = 10
                firstDay_2 = 2
                numDays_2 = 10
                periods = [[firstDay_1, numDays_1], [firstDay_2, numDays_2]]
                fileName = 'test'
                periodBlocks.append(periods)
                outputNames.append(fileName)
            for periodBlock, fileName in zip(periodBlocks, outputNames):
                selected_symbols = sorted(list(tse.symbols_data.all_symbols()))
                DL = DataLoader(selected_symbols, fileName=fileName, typeData=typeData, periods=periodBlock, MKDIR=False)
                DL.downloadCTD()
                DL.downloadHTD()
        
        elif _input == 2:
            typeData = 'client&history'

            print('\nNumber of Period Blocks: \n')
            nBlocks = int(input('> '))

            periodBlocks = []
            outputNames = []

            for i in range(nBlocks):
                print(f'\nBlock {i+1}')

                print('\nFirst Period\n')

                print('\nFirst Day: ')
                firstDay_1 = int(input('> '))

                print('\nNumber of Days: ')
                numDays_1 = int(input('> '))

                print('\nSecod Period\n')

                print('\nFirst Day: \n')
                firstDay_2 = int(input('> '))

                print('\nNumber of Days: \n')
                numDays_2 = int(input('> '))


                periods = [[firstDay_1, numDays_1]]
                if firstDay_2 != 0 and numDays_2 != 0:
                    periods = [[firstDay_1, numDays_1], [firstDay_2, numDays_2]]
                    

                print('\nEnter the file name: \n')
                fileName = str(input('> '))

                periodBlocks.append(periods)
                outputNames.append(fileName)

            for periodBlock, fileName in zip(periodBlocks, outputNames):
                selected_symbols = sorted(list(tse.symbols_data.all_symbols()))
                DL = DataLoader(selected_symbols, fileName=fileName, typeData=typeData, periods=periodBlock)
                DL()
            DL.wrapping(periodBlocks, outputNames)
        
        elif _input == 3:
            break
        else:
            raise ValueError('input is not valid.')
