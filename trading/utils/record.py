import os
import pandas as pd

from datetime import datetime

def save_predicted_result(predicted_result, confidence, model_type=None):
    # 確認 records/ 資料夾是否存在，不存在就建立資料夾
    target_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/')
    file_name = 'predicted_results.csv'

    if not os.path.exists(target_dir):
        print('Target directory does not exist. Creating new one...')
        os.mkdir(target_dir)

    csv_file = target_dir + file_name
    record_time = datetime.now()

    if not os.path.exists(csv_file):
        print('Target csv does not exist. Creating new one...')
        d = {'datetime': [record_time], 'predicted_result': [predicted_result], 'confidence': [confidence], 'model': [model_type]}
        df = pd.DataFrame(data=d)
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
        df_new = pd.DataFrame(data={'datetime': [record_time], 'predicted_result': [predicted_result], 'confidence': [confidence], 'model': [model_type]})
        df = pd.concat([df, df_new])
        df.to_csv(csv_file, index=False)

def save_position_history(symbol, action, entry_price=None, take_profit=None, stop_loss=None, trailing_stop_threshold=None, callback_rate=None, 
                          rsi_peak=None, no_keep_trend_threshold=None, entry_prediction=None, close_price=None, exit_condition=None):
    # 確認 records/ 資料夾是否存在，不存在就建立資料夾
    target_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/')
    if not os.path.exists(target_dir):
        print('Target directory does not exist')

    record_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_position_csv = target_dir + 'current_position.csv'
    position_history_csv = target_dir + 'position_history.csv'

    # 建立持倉時，紀錄 current position，結束持倉時，轉為紀錄至 position history
    if action in ['Long', 'Short']:
        # 確認 current position 紀錄檔是否存在
        # 不存在則建立紀錄檔，已存在則寫入紀錄檔
        if not os.path.exists(current_position_csv):
            print('save_position_history: Target csv does not exist')
            d = {'datetime': [record_time], 
                 'symbol': [symbol], 
                 'side': [action], 
                 'entry_price': [entry_price], 
                 'take_profit': [take_profit], 
                 'stop_loss': [stop_loss],
                 'trailing_stop_threshold': [trailing_stop_threshold],
                 'callback_rate': [callback_rate],
                 'hold_count': 0,
                 'loss_hold_count': 0,
                 'no_keep_update_max_profit_count': 0,
                 'no_keep_trend_threshold': [no_keep_trend_threshold] if no_keep_trend_threshold is not None else [[]],
                 'rsi_peak': [rsi_peak] if rsi_peak is not None else '',
                 'max_profit': 0, 
                 'entry_prediction': [entry_prediction]}
            df = pd.DataFrame(data=d)
            df.to_csv(current_position_csv, index=False)
        else:
            df = pd.read_csv(current_position_csv)
            if not df.empty:
                print('save_position_history: Currently holding position')
            else:
                d = {'datetime': [record_time], 
                     'symbol': [symbol], 
                     'side': [action], 
                     'entry_price': [entry_price], 
                     'take_profit': [take_profit], 
                     'stop_loss': [stop_loss],
                     'trailing_stop_threshold': [trailing_stop_threshold],
                     'callback_rate': [callback_rate],
                     'hold_count': 0, 
                     'loss_hold_count': 0,
                     'no_keep_update_max_profit_count': 0,
                     'no_keep_trend_threshold': [no_keep_trend_threshold] if no_keep_trend_threshold is not None else [[]],
                     'rsi_peak': [rsi_peak] if rsi_peak is not None else '',
                     'max_profit': 0, 
                     'entry_prediction': [entry_prediction]}
                df = pd.DataFrame(data=d)
                df.to_csv(current_position_csv, index=False)
    elif 'Exit' in action:
        if not os.path.exists(current_position_csv):
            print('save_position_history: Current position csv does not exist')
        else:
            # 讀取 entry position 資料
            df = pd.read_csv(current_position_csv)
            entry = df.iloc[0]
            
            # 若有歷史 position 資料，進行讀取    
            if not os.path.exists(position_history_csv):
                df_history = pd.DataFrame()
            else: 
                df_history = pd.read_csv(position_history_csv)

            # 計算 profit 與 close_price
            profit = None
            if close_price is not None and stop_loss is not None:
                if entry['side'] == 'Long':
                    profit = round((close_price - entry['entry_price']) / entry['entry_price'] * 100, 2)
                    if profit > 0 and take_profit is not None:
                        close_price = round(min(close_price, entry['entry_price'] * (1 + take_profit)), 2)
                    else:
                        close_price = round(max(close_price, entry['entry_price'] * (1 - stop_loss)), 2)
                elif entry['side'] == 'Short' :
                    profit = round((entry['entry_price'] - close_price) / entry['entry_price'] * 100, 2)
                    if profit > 0 and take_profit is not None:
                        close_price = round(max(close_price, entry['entry_price'] * (1 - take_profit)), 2)
                    else:
                        close_price = round(min(close_price, entry['entry_price'] * (1 + stop_loss)), 2)
                
                profit = max(profit, -(stop_loss * 100))
                if take_profit and take_profit != 0:
                    profit = min(profit, take_profit * 100)
            else:
                print('save_position_history: Close price or model sl is missing')

            # 產生與紀錄最新一筆 position 紀錄            
            d = {'symbol': [symbol], 
                 'side': [entry['side']], 
                 'profit': [profit] if profit is not None else '',
                 'max_runup': [entry['max_profit']],
                 'entry_price': [entry['entry_price']], 
                 'close_price': [close_price] if close_price is not None else '', 
                 'opened_at': [entry['datetime']], 
                 'closed_at': [record_time], 
                 'entry_prediction': [entry['entry_prediction']], 
                 'exit_condition': [exit_condition]
                }
            
            df_history_new = pd.DataFrame(data=d)

            # 將歷史與最新 position 合併後進行儲存
            df_history_combined = pd.concat([df_history, df_history_new])
            df_history_combined.to_csv(position_history_csv, index=False)

            # 清除 current position 紀錄
            df = df.drop([0])
            df.to_csv(current_position_csv, index=False)

            return df_history_new.iloc[-1].to_dict()
    elif action == 'Set v2 TP':
        position = pd.read_csv(current_position_csv)

        columns_to_update = ['take_profit', 'trailing_stop_threshold', 'callback_rate']
        # Convert column dtype to avoid incompatible dtype error
        position[columns_to_update] = position[columns_to_update].astype('float64')
        position.loc[0, columns_to_update] = [take_profit, trailing_stop_threshold, callback_rate]

        df = pd.DataFrame(data=position)
        df.to_csv(current_position_csv, index=False)

def check_current_position():
    position = 0
    df = pd.DataFrame()

    # 確認 records/ 資料夾是否存在
    target_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/')
    csv_file = ''
    
    if not os.path.exists(target_dir):
        print('check_current_position: Target directory does not exist')
    else:
        csv_file = target_dir + 'current_position.csv' 
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # 確認目前是否有持倉
        if df.empty:
            print('No current position')
        else:
            position = 1 if df.iloc[0]['side'] == 'Long' else 2
        return position, df              
    else:
        print('check_current_position: CSV file not found')
        return position, df
    