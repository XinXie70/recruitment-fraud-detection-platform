FROM node:22.22-alpine

WORKDIR /app

COPY --chown=node:node frontend/package*.json ./
RUN npm ci \
    && chown -R node:node /app/node_modules

COPY --chown=node:node frontend ./

USER node

EXPOSE 5190

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
