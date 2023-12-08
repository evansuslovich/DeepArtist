# Artelligence
Genre Classification

DeepArtistApp contains the web app.
DeepArtistApp/backend/deepartist contains the model and training data.

The web app uses the model that is trained in DeepArtistApp/backend/deepartist.


**To run backend:**
cd DeepArtistApp/backend

pip install flask torch

flask --app base.py --debug run

**To run frontend:**
cd DeepArtistApp/frontend

npm install

npm start
