# Auth (Removed)

Authentication and authorization have been removed so the app focuses only on the prediction page.

## Previous Auth Flow (For Reference)
The earlier implementation used:
- SQLite database for user storage.
- Argon2 password hashing via `passlib`.
- JWTs stored in an `access_token` cookie.
- Role-based access with `admin` and `user` roles.

### User Signup
`POST /signup` accepted:
- `first_name`, `last_name`, `phone`, `email`, `password`, `confirm_password`

Steps:
1. Validate required fields and password confirmation.
2. Hash password (argon2).
3. Store user with role `user`.
4. Redirect to `/login?created=1`.

### Login
`POST /login` accepted:
- `email`, `password`

Steps:
1. Validate user credentials.
2. Issue JWT (`sub` set to user id).
3. Set `access_token` cookie (HTTP-only).

### Admin
Admin-only endpoints:
- `GET /admin` (admin UI page)
- `GET /api/admin/users` (list users)
- `POST /api/admin/users/{id}/role` (set role)
- `POST /api/admin/users/{id}` with `method=delete` (delete user)

### Middleware/Checks
On each protected route:
1. Read `access_token` cookie.
2. Verify JWT.
3. Load user from DB.
4. Check role for admin routes.
