import { z } from 'zod';

export const environmentSchema = z.object({
  VITE_API_BASE_URL: z
    .url()
    .refine((value) => {
      try {
        const url = new URL(value);
        return (
          ['http:', 'https:'].includes(url.protocol) &&
          !url.username &&
          !url.password &&
          !url.search &&
          !url.hash &&
          url.pathname === '/'
        );
      } catch {
        return false;
      }
    }, 'VITE_API_BASE_URL must be an HTTP(S) origin without credentials, query or fragment')
    .default('http://localhost:8000'),
});
