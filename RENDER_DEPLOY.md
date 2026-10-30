# Deploy ChurnQ on Render

1. Put this project folder in a GitHub repository. The active model files in
   `processed/active_model_*` are allowed by `.gitignore` and should be
   committed; uploaded datasets and per-upload outputs remain excluded.
2. In Render, create a Blueprint from that repository and select this
   folder's `render.yaml` if the repository root is a parent folder.
3. Set `GEMINI_API_KEY` in the Render service environment using a newly
   generated key. The key previously stored in `config.py` should be revoked.
   Render generates `FLASK_SECRET_KEY` from the Blueprint.
4. Deploy. Render will install `requirements-render.txt` and start the Flask
   app with Gunicorn. The chatbot remains hosted separately on Netlify and is
   embedded by the ChurnQ app.

The Blueprint uses Render's free instance. Its filesystem is ephemeral, so
uploaded datasets and generated per-upload files may be lost after a restart
or redeploy. The active model bundle is included with the app source.