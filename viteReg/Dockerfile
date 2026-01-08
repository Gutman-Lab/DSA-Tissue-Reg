# Multi-stage build for Vite React app
FROM node:20-alpine AS builder

# Install git (needed for GitHub dependencies)
RUN apk add --no-cache git

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including bdsa-react-components from npm)
# Note: bdsa-react-components has a peer dependency from GitHub that requires git
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

