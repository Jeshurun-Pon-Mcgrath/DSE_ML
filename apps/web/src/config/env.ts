import { environmentSchema } from './environment-schema';

export const env = Object.freeze(environmentSchema.parse(import.meta.env));
