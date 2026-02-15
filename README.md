# Dorm Pass Manager

Dorm Pass Manager is a Flask-based web application for managing student dorm passes, approvals, and KIOSK/mobile sign-ins. It provides role-based access (admin / proctor / approver), student and user management, location handling, and optional face/ID verification features for KIOSK mode.

## Features

- Pass creation, multi-stage approval, and elapsed-time warnings.
- Role-based access control (numeric roles: 1 = admin, 2 = proctor, 3 = approver).
- KIOSK mode with PIN and optional face verification flows.
- User and student management (add / edit / delete).
- Location / destination management.
- Real-time updates via Socket.IO.
- Configurable settings via `config.json`.

## Tech Stack

- Python + Flask
- Flask-SocketIO
- Vanilla JavaScript for front-end logic (`templates/` and `static/js/`)
- MySQL for persistent storage (accessed via `models/` wrappers)
- DeepFace / face_recognition (optional) for face compare
- Pillow / OpenCV for image processing

## Prerequisites

- Python 3.10+ (recommended)
- MySQL server (or remote MySQL DB)
- `python3`, `pip`
- (Optional for production) `gunicorn` and `eventlet`/`gevent`

## Quick Start (Local Development)

1. Clone the repo and change into it:

```bash
git clone <repo-url>
cd Dorm-Pass-Manager
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure `config.json`:

- Ensure `config.json` contains the correct MySQL connection, SMTP (if used), session and captcha settings, and campus metadata.

5. Run the app (development):

```bash
python app.py
```

Note: `app.py` defaults to `socketio.run(app, port=80, host="0.0.0.0", debug=...)`. Running on port 80 requires elevated privileges; for local development consider changing the port to 5000 or running with sudo (not recommended).

6. Production-style run (Socket.IO-compatible):

Install `eventlet`:

```bash
pip install eventlet
```

Run with Gunicorn (example):

```bash
gunicorn -k eventlet -w 1 wsgi:app
```

Use a reverse proxy (Nginx) in front of Gunicorn for TLS/port 80/443 management in production.

## Project Structure (high level)

- `app.py` — main Flask app and API routes (entrypoint uses `socketio.run`).
- `wsgi.py` — WSGI wrapper for server deployments.
- `models/` — DB access and helper modules.
- `templates/` — HTML templates for pages and admin panels.
- `static/` — JS, CSS, and face model assets.
- `config.json` — application configuration.
- `requirements.txt` — Python dependencies.

## Configuration

- Edit `config.json` to set the database credentials, session options, captcha settings, SMTP credentials, and campus information.

## Troubleshooting

- Port 80 permission errors: run using a non-privileged port or use a reverse proxy.
- Database connection errors: verify values in `config.json` and that MySQL is reachable.
- Missing Python packages: activate the virtualenv and run `pip install -r requirements.txt`.
- Socket.IO issues: use an `eventlet` or `gevent` worker in production.
- Face recognition: native dependencies like OpenCV/dlib may require platform-specific installation; consider Docker for consistent environments.

## Development Tips

- Use a virtualenv and keep `requirements.txt` up to date.
- Add a `config.sample.json` with placeholders to help onboarding.
- Consider adding Docker/Docker Compose for reproducible development and CI.

## Next Improvements

- Add DB migrations / seed scripts for easy setup.
- Provide a sample `config.sample.json` and example data.
- Add automated tests and CI workflow.
- Add a `Dockerfile` and `docker-compose.yml` for easy local deployment.

---

If you want, I can also add a `config.sample.json`, a brief Dockerfile, or create initial DB migration scripts — tell me which you'd like next.
