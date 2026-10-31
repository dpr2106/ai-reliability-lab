# syntax=docker/dockerfile:1
FROM node:22-alpine AS builder

WORKDIR /app

# Reproducible installation from lockfile
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Published backend API URL for browser-side requests
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=${VITE_API_URL}

# Run static type checks, unit tests, and production build
RUN npm run check && npm test && npm run build

# Prune development dependencies for minimal runtime footprint
RUN npm prune --production

# -----------------------------------------------------------------------------
# Runtime Stage: minimal production image
# -----------------------------------------------------------------------------
FROM node:22-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=5173

COPY --chown=node:node --from=builder /app/package.json ./
COPY --chown=node:node --from=builder /app/node_modules ./node_modules
COPY --chown=node:node --from=builder /app/build ./build

USER node

EXPOSE 5173

CMD ["node", "build/index.js"]

