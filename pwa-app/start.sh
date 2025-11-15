#!/bin/bash

# Options Flow Pro - Quick Start Script
# This script sets up and starts the PWA application

set -e  # Exit on error

echo "🚀 Options Flow Pro - Setup & Start"
echo "===================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the pwa-app directory"
    exit 1
fi

# Backend Setup
echo "📦 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

# Frontend Setup
echo "📦 Setting up frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies (this may take a few minutes)..."
    npm install --silent
else
    echo "Dependencies already installed"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_VAPID_PUBLIC_KEY=
EOL
    echo "⚠️  Remember to add your VAPID key to .env for push notifications"
fi

echo "✅ Frontend setup complete!"
echo ""

# Start services
echo "🚀 Starting services..."
echo ""

# Start backend in background
cd ../backend
echo "Starting backend API on http://localhost:8000..."
source venv/bin/activate
python main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check logs/backend.log"
    exit 1
fi

# Start frontend
cd ../frontend
echo ""
echo "Starting frontend on http://localhost:3000..."
echo ""
echo "========================================="
echo "🎉 Application is starting!"
echo "========================================="
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start frontend (this will block)
npm start

# Cleanup on exit
trap "echo 'Stopping services...'; kill $BACKEND_PID 2>/dev/null; exit" INT TERM
