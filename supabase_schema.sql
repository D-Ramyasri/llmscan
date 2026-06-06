create table if not exists public.users (
    id serial primary key,
    email text not null unique,
    password_hash text not null,
    state jsonb not null default '{}'::jsonb,
    created_at timestamp not null default now(),
    updated_at timestamp not null default now()
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_users_updated_at on public.users;

create trigger set_users_updated_at
before update on public.users
for each row
execute function public.set_updated_at();
