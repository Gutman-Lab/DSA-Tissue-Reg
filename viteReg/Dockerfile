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

# Accept build arguments for user ID
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create nginx cache directories and pid file location with proper permissions
RUN mkdir -p /var/cache/nginx/client_temp \
    /var/cache/nginx/proxy_temp \
    /var/cache/nginx/fastcgi_temp \
    /var/cache/nginx/uwsgi_temp \
    /var/cache/nginx/scgi_temp \
    /var/run/nginx \
    /run && \
    chown -R nginx:nginx /var/cache/nginx /var/run/nginx /run && \
    chmod -R 755 /var/cache/nginx /var/run/nginx && \
    chmod 1777 /run

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configurations
COPY nginx-main.conf /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user if USER_ID is not 0 (root)
RUN if [ "$USER_ID" != "0" ]; then \
    deluser nginx 2>/dev/null || true && \
    addgroup -g $GROUP_ID nginx 2>/dev/null || true && \
    adduser -D -u $USER_ID -G nginx nginx 2>/dev/null || true && \
    chown -R nginx:nginx /usr/share/nginx/html /var/cache/nginx /var/run/nginx /run; \
    fi

EXPOSE 80

# Run nginx as the specified user (or default nginx user)
USER ${USER_ID:-nginx}

CMD ["nginx", "-g", "daemon off;"]

