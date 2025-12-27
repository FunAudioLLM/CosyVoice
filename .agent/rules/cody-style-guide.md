---
trigger: always_on
---

# Antigravity Rules: Project Coding Standards & Patterns

These rules are *always-on* constraints and coding patterns for this repo. They define **how code should look and be changed** (procedural steps belong in workflows).

## General Engineering Rules

- **Separation of concerns:** keep modules small and single-purpose; do not mix UI, data-access, and orchestration.
- **Keep diffs tight:** prefer focused commits; avoid drive-by refactors unless required.
- **No “magic edits”:** if you change behavior, update tests/docs/logs accordingly.
- **Bulk edits (>5 files):** template first; apply in 2–3 file batches; run `tsc --noEmit` after each batch.
- **Package removal safety:** run `pnpm why <pkg>` first; after removal run `pnpm install && pnpm lint`.

## Tooling & Edit Strategy

### Prefer the right edit primitive
- Whole-file replace: `write_to_file(Overwrite: true)`
- 1–3 localized changes: `replace_file_content`
- Many non-contiguous edits: `multi_replace_file_content`

### Package installation verification
Editing `package.json` does not guarantee binaries exist—use explicit installs:
```bash
pnpm add -D <pkg>
```

### Linter migrations (ESLint → Biome → oxlint, etc.)
1. Update `package.json` scripts
2. Update/remove config files
3. Rewrite inline disable comments:
   ```bash
   grep -r "eslint-disable\|biome-ignore" src/ --include="*.ts" --include="*.tsx"
   ```
4. Remove old + add new packages
5. Run lint to verify

### “Missing linter” quick fix
```bash
pnpm add -D oxlint
```

## TypeScript / React Patterns

### Avoid `never[]` inference
```ts
// BAD
const items = []
// GOOD
const items: ItemType[] = []
```

### Map tuple inference
```ts
// BAD
const map = new Map(items.map(item => [item.id, item]))
// GOOD
const map = new Map(items.map(item => [item.id, item] as const))
```

### React Query refactors (state/effect → query)
1. Change function signatures in one edit.
2. Update **all** call sites in the same file.
3. Ensure `useCallback` deps are correct.
4. Run `tsc --noEmit` after major hook refactors.

### Memoize list rows for large renders
- Extract a row component and `memo()` it.
- Use `useCallback` for row handlers.

### Shadcn/UI “purity” rules
If `Math.random()` in render/memo triggers purity linting, replace with a deterministic value or move generation outside render.

## Drizzle ORM

### Dynamic `where` composition
```ts
import { SQL, eq, and } from "drizzle-orm"

const conditions: SQL<unknown>[] = []
if (filter) conditions.push(eq(table.column, value))

const results = await db.query.table.findMany({
  where: conditions.length ? and(...conditions) : undefined,
})
```

## Styling & UI Architecture

### CSS Modules migration rules
1. Create `Component.module.css` adjacent to the component.
2. Use standard CSS properties.
3. Import as `styles` and compose with `cn()`.
4. Prefer attribute selectors for state.

### Layout shell pattern for persistent nav
- Create a `LayoutShell` wrapping providers + navigation.
- In `layout.tsx`, wrap `{children}` in `LayoutShell`.

## Auth Modeling

- Match the actual `User` interface shape; do not invent fields.

## Logging & Documentation Standards

### Runtime log prefixes
Use consistent prefixes for grep/filtering: `[API]`, `[DB]`, `[Query]`, `[Hook]`, `[Cache]`, `[PERF]`.

### JSDoc
**New exported files/APIs:**
- Add `@fileoverview` and module description.
- Add JSDoc to exported functions/classes.
- Include `@example` for non-obvious usage.

**Modifying behavior:**
- Update JSDoc; add `// PERF:` / `// FIX:` notes where relevant.

## Changelog

```md
## [Unreleased] - YYYY-MM-DD

### Added
### Changed
### Fixed
### Performance
```
