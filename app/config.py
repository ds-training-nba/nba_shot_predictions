

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

TARGET_VARIABLE = "SHOT_MADE_FLAG"

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
    "SHOT_DISTANCE",
    "SHOT_ZONE_RANGE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "LOC_X",
    "LOC_Y",
    "ACTION_TYPE",
    "HTM",
    "PLAYER1_TEAM_ABBREVIATION",
    "PLAYER_ID",
    "scoreHome",
    "scoreAway",
    TARGET_VARIABLE,
]

EXPLANATORY_VARIABLES_PRE_ENCODING = [
    "PLAYER_ID",
    "IS_HOME",
]


