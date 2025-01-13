import os, json

from requests import Session
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

trading_hosts = json.loads(os.getenv('TRADING_HOSTS'))

server_account = os.getenv('SERVER_ACCOUNT')
server_password = os.getenv('SERVER_PASSWORD')

def generate_trade_request_body(symbol, order_type, model_type, sl_value=None, sl_price=None, tp_type=None, tp_value=None, trailing_activation=None, 
                                trailing_offset=None, runup_stop_price=None, current_price=None):
    body = {
        'ticker': symbol,
        'order': order_type,
        'model': model_type,
        'time': datetime.now()
    }

    if sl_value is not None:
        body['sl_value'] = sl_value

    if sl_price is not None:
        body['sl_price'] = round(sl_price, 2)

    if 'TP' in order_type:
        body['tp_type'] = tp_type if tp_type else 'Percentage'
        if tp_value is not None:
            body['tp_value'] = tp_value

    if trailing_activation is not None and trailing_offset is not None:
        body['trailing_activation'] = trailing_activation
        body['trailing_offset'] = trailing_offset

    if runup_stop_price is not None:
        body['runup_stop_price'] = runup_stop_price

    if current_price is not None:
        body['current_price'] = current_price

    return body

def send_request(trading_host='', api_path='', method='', body={}, bearer_token=None, query_params={}):
    session = Session()

    response = {}
    params = {
        'url': trading_host + api_path,
        'data': json.dumps(body, default=str),
        'params': query_params
    }
    print(f'Request params: {params}')

    session.headers.update(
        {
            'Content-Type': 'application/json;charset=utf-8'
        }
    )

    if bearer_token:
        session.headers.update({'Authorization': f'Bearer {bearer_token}'})

    if method == 'post':
        response = session.post(**params)
    elif method == 'get':
        response = session.get(**params)

    try:
        response = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        print('Response can not be converted to JSON')
        print('Please check trading server for error message')

    print(f'[Response]\n{response}')

    return response

def send_order(symbol, order, model_type, sl_value=None, sl_price=None, tp_type=None, tp_value=None, trailing_activation=None, trailing_offset=None, 
               runup_stop_price=None, current_price=None):
    if symbol and order and model_type:
        body = generate_trade_request_body(symbol, f'{order}', model_type, sl_value, sl_price, tp_type, tp_value, trailing_activation, trailing_offset, 
                                           runup_stop_price, current_price)
        for trading_host in trading_hosts:
            send_request(trading_host, '/webhook', 'post', body)
    else:
        print('send_order: Missing required parameters (symbol, order, model_type)')

def login_server(host=None):
    server_token = {}
    login_failure = []

    if host:
        response = send_request(host, '/users/login/', 'post', body={'username': server_account, 'password': server_password})

        if response.get('access') is not None:
            server_token[host] = response['access']
        else:
            server_token[host] = ''
            login_failure.append(host)
    else:
        if len(trading_hosts) == 0:
            print('login_server: No trading hosts set')
            return None

        for trading_host in trading_hosts:
            response = send_request(trading_host, '/users/login/', 'post', body={'username': server_account, 'password': server_password})

            if response.get('access') is not None:
                server_token[trading_host] = response['access']
            else:
                server_token[trading_host] = ''
                login_failure.append(trading_host)

    if login_failure:
        print(f'login_server: Failed to get JWT token from {login_failure}')

    print(f'login_server: JWT token obtained from {server_token.keys()}')
    return server_token

def send_trading_record(data=None, model_type=None, token=None):
    if data is None or not isinstance(data, dict):
        print('send_trading_record: Invalid data type')
        return

    if model_type is None:
        print('send_trading_record: Missing model type')
        return

    data['model'] = model_type

    if len(trading_hosts) == 0:
        print('send_trading_record: No trading hosts set')
        return None

    for trading_host in trading_hosts:
        if token is None:
            server_token = login_server(trading_host)
            access_token = server_token.get(trading_host)
        else:
            access_token = token.get(trading_host)

        if access_token:
            send_request(trading_host, '/strategy-records/', 'post', body=data, bearer_token=access_token)

    return

# Get a user who is currently using the specified model
def get_strategy_user(model=None, token=None):
    strategy_user = {}

    if len(trading_hosts) == 0:
        print('get_strategy_user: No trading hosts set')
        return None

    if not model:
        print('get_strategy_user: Missing model name')
        return None

    for trading_host in trading_hosts:
        if token is None:
            server_token = login_server(trading_host)
            access_token = server_token.get(trading_host)
        else:
            access_token = token.get(trading_host)

        if access_token:
            response = send_request(trading_host, '/strategies/query-users/', 'get', bearer_token=access_token, query_params={'strategy_name': model})

            if response is not None and 'users' in response:
                if trading_host == 'http://52.64.130.38':
                    if 2 in response['users']:
                        strategy_user[trading_host] = 2
                    else:
                        strategy_user[trading_host] = response['users'][0]
                elif trading_host == 'http://13.239.157.51':
                    if 4 in response['users']:
                        strategy_user[trading_host] = 4
                    else:
                        strategy_user[trading_host] = response['users'][0]
                else:
                    strategy_user[trading_host] = response['users'][0]

                print(f'get_strategy_user: Found user {strategy_user[trading_host]} for the specified model {model} on {trading_host}')
                return strategy_user
            else:
                print(f'get_strategy_user: No user found for the specified model {model} on {trading_host}')

    return None

def query_user_order(strategy_user=None, symbol=None, order_id=None, token=None):
    if strategy_user is None:
        print('query_user_order: Missing strategy user')
        return None

    if not isinstance(strategy_user, dict):
        print('query_user_order: Invalid strategy user type')
        return None

    for key, value in strategy_user.items():
        trading_host = key
        user_id = value

        if token is None:
            server_token = login_server(trading_host)
            access_token = server_token.get(trading_host)
        else:
            access_token = token.get(trading_host)

        if access_token:
            response = send_request(trading_host, f'/users/query-order/{user_id}/', 'get', bearer_token=access_token, query_params={'symbol': symbol, 'client_id': order_id})

            return response

    return None

def query_user_trade(strategy_user=None, symbol=None, token=None):
    if strategy_user is None:
        print('query_user_trade: Missing strategy user')
        return None

    if not isinstance(strategy_user, dict):
        print('query_user_trade: Invalid strategy user type')
        return None

    for key, value in strategy_user.items():
        trading_host = key
        user_id = value

        if token is None:
            server_token = login_server(trading_host)
            access_token = server_token.get(trading_host)
        else:
            access_token = token.get(trading_host)

        if access_token:
            response = send_request(trading_host, f'/users/query-trade/{user_id}/', 'get', bearer_token=access_token, query_params={'symbol': symbol})

            return response

    return None

def check_server_time():
    response = send_request('https://fapi.binance.com', '/fapi/v1/time', 'get')

    if response and 'serverTime' in response:
        return response['serverTime']
    else:
        print('check_server_time: Failed to get server time')
        return None
