from selenium.webdriver import Chrome
from selenium.webdriver import Safari
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import tqdm
import pathlib


def open_page(gender):
    url = f'https://en.volleyballworld.com/volleyball/world-ranking/{gender}'

    # Make sure the folder where we'll save the data exists
    folder = pathlib.Path('../DATA/FIVB') / gender
    folder.mkdir(parents=True, exist_ok=True)

    # Column names (for the CSV)
    columns = ['dates', 'home_worldranking', 'home_team', 'home_results',
               'away_worldranking', 'away_team', 'away_results', 'increment']

    # List of information per game
    games_list = []

    # Open Google Chrome Driver
    with Chrome() as driver:
    # with Safari() as driver:
        # Open website
        driver.get(url)
        # driver.maximize_window()

        # Make clicks wait for the element to load (max 20s)
        wait = WebDriverWait(driver, 20)

        # Accept cookies
        elem = (By.ID, 'onetrust-accept-btn-handler')
        wait.until(EC.element_to_be_clickable(elem)).click()

        # Load all teams
        elem = (By.CLASS_NAME, 'load-more-btn')
        wait.until(EC.element_to_be_clickable(elem)).click()

        # Find table
        table = driver.find_element(By.ID, 'vbw-o-table-wr-outer')
        table_items = table.find_elements(
            By.CLASS_NAME, 'vbw-ranking-page-table-body')

        # Iterate over teams
        for item in tqdm.tqdm(table_items):

            # Find the table of games for that team
            print(item.text.split()[:2])
            subtable = item.find_element(By.CLASS_NAME, 'table-hidden-row')
            team_games = subtable.find_elements(By.CLASS_NAME, 'vbw-o-table__row')

            # Iterate over games
            for game in tqdm.tqdm(team_games, leave=False):
                # Initialize dictionary with the correct keys (in case some are missing in the webpage)
                game_dict = dict(zip(columns, [None] * len(columns)))

                # Init a counter for the 'worldranking' key (which appears twice, but in order: 1-home, 2-away)
                wr = 0

                # Extract all the information for this game
                game_info = game.find_elements(By.CLASS_NAME, 'vbw-o-table__cell')
                for info in game_info:
                    data = info.get_attribute('innerHTML')
                    info_name = info.get_attribute('class').split()[1]
                    if info_name in ['trend', 'empty-one']:
                        continue
                    elif info_name == 'worldranking':
                        wr += 1
                        if wr == 1:
                            game_dict['home_team'] = data
                        elif wr == 2:
                            game_dict['away_team'] = data
                        else:
                            print(f'`{info_name}` appered {wr} times (!!!???)')
                            continue
                    elif info_name == 'dates':
                        data = info.find_element(By.CLASS_NAME, 'dates-short').get_attribute('innerHTML')
                        game_dict['dates'] = pd.to_datetime(data)
                    elif info_name == 'home':
                        game_dict['home_worldranking'] = data
                    elif info_name == 'away':
                        game_dict['away_worldranking'] = data
                    elif info_name == 'results':
                        raw_data = data.split()
                        game_dict['home_results'] = int(raw_data[0])
                        game_dict['away_results'] = int(raw_data[-1])
                    elif info_name == 'increment':
                        game_dict['increment'] = abs(float(data))

                # Save it for later
                games_list.append(game_dict)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(games_list)
    df = df.drop_duplicates().sort_values(by='dates').reset_index(drop=True)

    # Save to CSV
    df.to_csv(folder / 'data.csv', index=False)

    return df

############################################################
############################################################

df = open_page('men')
df = open_page('women')

# folder = pathlib.Path('volley/men')
# df = pd.read_csv(folder / 'data.csv')
# df.to_csv(folder / 'data2.csv', index=False)


print()
