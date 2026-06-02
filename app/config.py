# App definitions

RESULTS_PATH = 'doc/results'
EXPERIMENTS_PATH = RESULTS_PATH + '/experiments'

# Candidates to choose from
ESPN_TOP_25 = [
    'LeBron James',
    'Kobe Bryant',
    'Stephen Curry',
    'Tim Duncan',
    "Shaquille O'Neal",
    "Kevin Garnett",
    "Nikola Jokic",
    "Dwyane Wade",
    "Kevin Durant",
    "Dirk Nowitzki",
    "Giannis Antetokounmpo",
    "Steve Nash",
    "James Harden",
    "Jason Kidd",
    "Chris Paul",
    "Kawhi Leonard",
    "Manu Ginobili",
    "Allen Iverson",
    "Anthony Davis",
    "Ray Allen",
    "Tony Parker",
    "Draymond Green",
    "Russell Westbrook",
    "Pau Gasol",
    "Luka Doncic"
]

# our actual choice from our selection algorithm
# (Take list of 25 and sort by last match, so we get the most current 20 of the list)
PLAYER_CHOICE = [
    'LeBron James',
    'Kobe Bryant',
    'Stephen Curry',
    'Tim Duncan',
 #   "Shaquille O'Neal",
    "Kevin Garnett",
    "Nikola Jokic",
    "Dwyane Wade",
    "Kevin Durant",
    "Dirk Nowitzki",
    "Giannis Antetokounmpo",
 #   "Steve Nash",
    "James Harden",
  #  "Jason Kidd",
    "Chris Paul",
    "Kawhi Leonard",
    "Manu Ginobili",
   # "Allen Iverson",
    "Anthony Davis",
  #  "Ray Allen",
    "Tony Parker",
    "Draymond Green",
    "Russell Westbrook",
    "Pau Gasol",
    "Luka Doncic"
]

# For convenience and consitency
TARGET_VARIABLE = "SHOT_MADE_FLAG"

# For some first experiments
EXPLANATORY_CANDIDATES_NUMERICAL = [
    "SHOT_DISTANCE",
    "PERIOD_x",
    "MINUTES_REMAINING",
    "ANGLE",
    "ABS_ANGLE",
    "ANGLE_SECTOR",
    "IS_HOME",
    "OPPONENT_INTERFERED"
]

# These are the columns that must not contain missing values, because they are taken as a source for our
# features
CLEAN_SOURCE_COLUMNS = [
    "GAME_ID_x",
    "GAME_EVENT_ID",
    "SHOT_TYPE",
    "SHOT_DISTANCE",
    "SHOT_ZONE_RANGE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "LOC_X",
    "LOC_Y",
    "ACTION_TYPE",
    "HTM",
    "VTM",
    "GAME_DATE",
    "TEAM_ID",
    "PLAYER1_TEAM_ABBREVIATION", # for IS_HOME
    "PLAYER_ID",
    "scoreHome", # for score margin
    "scoreAway",
    'shotResult', # consistency check
    'MINUTES_REMAINING',
    'SECONDS_REMAINING',
    'clock',
    "PLAYER_NAME",
    "TEAM_ID",
    "PERIOD_x",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "GAME_DATE",
    "PLAYER2_ID",
    "PLAYER2_NAME",
    "PLAYER2_TEAM_ABBREVIATION",
    "PLAYER3_ID",
    "PLAYER3_NAME",
    "is_playoffs",
    "PCTIMESTRING",
    TARGET_VARIABLE,
]

# legacy, now dynamically configured in EncodingConfig
EXPLANATORY_VARIABLES_PRE_ENCODING = [
    "PLAYER_ID",
    "MAIN_ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_DISTANCE",
    "ANGLE_SECTOR",
    "IS_HOME",
    "is_playoffs",
]


