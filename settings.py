## Do all login and DB stuff in here to make imports not a nightmare...
import girder_client, os
from os.path import join, dirname
from dotenv import load_dotenv
from colorama import Fore, Style
from joblib import Memory
import dash
import dash_bootstrap_components as dbc

# Imports that we might need later.
# import pymongo
# import socket
memory = Memory(".npCacheDir", verbose=0)

# Assuming that this file, settings.py, is in the same directory as the .env file.
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

DSAKEY = os.getenv("DSAKEY")
DSA_BASE_URL = os.getenv("DSA_BASE_URL")

USER = None
USER_IS_ADMIN = None

# Authenticate girder client.
gc = girder_client.GirderClient(apiUrl=DSA_BASE_URL)
if DSAKEY:
    try:
        response = gc.authenticate(apiKey=DSAKEY)
        USER = gc.getUser(response["_id"])["login"]  # user name

        # Get the information from current token.
        token_info = gc.get("token/current")

        # Find the user ID that owns the token.
        try:
            for user_info in token_info["access"]["users"]:
                user = gc.getUser(user_info["id"])
                USER = user["login"]
                USER_IS_ADMIN = user["admin"]
                break
        except KeyError:
            USER = "Could not match API token to user."
            USER_IS_ADMIN = False

    except:
        print(
            Fore.RED
            + "Could not authenticate girder client with given DSAKEY in .env file."
            + Style.RESET_ALL
        )
else:
    print(
        Fore.YELLOW
        + "No DSAKEY in .env file, can't authenticate girder client."
        + Style.RESET_ALL
    )


# Hard-coded for now, this is the folder of images we are developing the app around.
sampleFolderId = "65ae7d301691254b2fc77c2b"


## This creates a single dash instance that I can access from multiple modules
class SingletonDashApp:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SingletonDashApp, cls).__new__(cls)

            # Initialize your Dash app here
            cls._instance.app = dash.Dash(
                __name__,
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    dbc.icons.FONT_AWESOME,
                ],
                title="NeuroTK Dashboard",
                # long_callback_manager=lcm,
            )
        return cls._instance


COLORS = {
    "banner-main": "#002878",
    "bn-background": "#6384c6",
    "bn-font": "#fcfcfc",
    "background": "#e4e6f0",
    "background-secondary": "#ecedf0",
}


## MONGO CONNECTION INFORMATION

## Determine if I can connect to mongo and redis
## NOT USING REDIS ANYMORE... WILL MAYBE REINTEGRATE LATER
# try:
#     redis_host = socket.gethostbyname("redis")
#     # print(redis_host)
#     REDIS_URL = "redis://redis:6379"

# except:
#     print("Host lookup failed for REDIS")
#     REDIS_URL = "redis://localhost:6379"


# MONGO_URI = "localhost:37017"
# MONGODB_USERNAME = "docker"
# MONGODB_PASSWORD = "docker"
# MONGODB_HOST = "mongodb"
# MONGODB_PORT = 27017
# MONGODB_DB = "cacheDSAannotationData"
# APP_IN_DOCKER = False


# ### Create mongo connection strings here as well
# mongoConn = pymongo.MongoClient(
#     MONGO_URI, username=MONGODB_USERNAME, password=MONGODB_PASSWORD
# )
# dbConn = mongoConn[MONGODB_DB]

# dbConn["annotationData"].create_index("annotation.name")
# dbConn["annotationData"].create_index("itemId")

# dbConn["annotationData"].create_index([("itemId", 1), ("annotation.name", 1)])
# dbConn["imageTileInfo"].create_index("imageId")

# # imageReg_dbConn = mongoConn["registationDB"]
# # DSA settings.
# import os, girder_client
# from dotenv import load_dotenv
# from pathlib import Path
# import pymongo

# import dash
# import diskcache
# from dash.long_callback import DiskcacheLongCallbackManager
# import dash_bootstrap_components as dbc


# ## This creates a single dash instance that I can access from multiple modules
# class SingletonDashApp:
#     _instance = None

#     def __new__(cls):
#         if not cls._instance:
#             cls._instance = super(SingletonDashApp, cls).__new__(cls)

#             # Initialize your Dash app here
#             cls._instance.app = dash.Dash(
#                 __name__,
#                 external_stylesheets=[
#                     dbc.themes.BOOTSTRAP,
#                     dbc.icons.FONT_AWESOME,
#                 ],
#                 title="NeuroTK Dashboard",
#                 long_callback_manager=lcm,
#             )
#         return cls._instance


# cache = diskcache.Cache("./src/neurotk-cache")
# lcm = DiskcacheLongCallbackManager(cache)

# cache = diskcache.Cache("./neurotk-cache-directory")
# background_callback_manager = dash.DiskcacheManager(cache)


# # Load .env variables to environment.
# load_dotenv(dotenv_path=".env", override=True)


# def is_docker():
#     """
#     Adding code that if I am not running in a docker environment, it will use
#     different MONGO_DB Credentials.
#     """
#     cgroup = Path("/proc/self/cgroup")
#     return (
#         Path("/.dockerenv").is_file()
#         or cgroup.is_file()
#         and "docker" in cgroup.read_text()
#     )


# print(os.environ)
# APP_HOST = os.environ.get("HOST")
# APP_PORT = int(os.environ.get("PORT", 5000))
# DEV_TOOLS_PROPS_CHECK = bool(os.environ.get("DEV_TOOLS_PROPS_CHECK"))
# API_KEY = os.environ.get("API_KEY", None)
# DSA_BASE_URL = os.environ.get("DSA_BASE_URL", None)
# ROOT_FOLDER_ID = os.environ.get("ROOT_FOLDER_ID", None)
# ROOT_FOLDER_TYPE = os.environ.get("ROOT_FOLDER_TYPE", None)

# PROJECTS_ROOT_FOLDER_ID = os.environ.get(
#     "PROJECTS_ROOT_FOLDER_ID", "64dbd2667920606b462e5b83"
# )


# if is_docker():
#     MONGO_URI = os.environ.get("MONGO_URI", None)

#     MONGODB_USERNAME = os.environ.get("MONGODB_USERNAME", "docker")
#     MONGODB_PASSWORD = os.environ.get("MONGODB_PASSWORD", "docker")
#     MONGODB_HOST = os.environ.get("MONGODB_HOST", "mongodb")
#     MONGODB_PORT = os.environ.get("MONGODB_PORT", 27017)
#     MONGODB_DB = os.environ.get("MONGODB_DB", "dsaCache")
#     APP_IN_DOCKER = True
# elif os.environ.get("WINDOWS", None):
#     MONGO_URI = "localhost"
#     MONGODB_USERNAME = "docker"
#     MONGODB_PASSWORD = "docker"
#     MONGODB_HOST = "localhost"
#     MONGODB_PORT = 27017
#     MONGODB_DB = "dsaCache"
#     APP_IN_DOCKER = False
# else:
#     MONGO_URI = "localhost"
#     MONGODB_USERNAME = None
#     MONGODB_PASSWORD = None
#     MONGODB_HOST = "localhost"
#     MONGODB_PORT = 27017
#     MONGODB_DB = "dsaCache"
#     APP_IN_DOCKER = False


# AVAILABLE_CLI_TASKS = {
#     "PositivePixelCount": {"name": "Positive Pixel Count", "dsa_name": "PPC"},
#     "TissueSegmentation": {
#         "name": "TissueSegmentation",
#         "dsa_name": "TissueSegmentation",
#     },
#     "TissueSegmentationV2": {
#         "name": "TissueSegmentationV2",
#         "dsa_name": "TissueSegmentationV2",
#     },
#     "NFTDetection": {"name": "NFTDetection", "dsa_name": "NFTDetection"},
# }

# ## Move database connection to here
# mongoConn = pymongo.MongoClient(
#     MONGO_URI, username=MONGODB_USERNAME, password=MONGODB_PASSWORD
# )
# dbConn = mongoConn[
#     MONGODB_DB
# ]  ### Attach the mongo client object to the database I want to store everything
# ## dbConn is a connection to the dsaCache database
