# Promotions & Events

*When a product qualifies for multiple Amazon promotional events at the same time, the logic behind this script picks the most important one - making sure major shopping events like Prime Day or Black Friday take priority over smaller promotional events. This keeps our reporting clean and helps us better understand how different products perform during important shopping periods.*


## 1. Event Priority Definition

* Assigns numerical priority to each event type
* Lower numbers = higher priority (Prime Day is highest priority at 1)
* All other events get lowest priority (99)

```
WITH event_priority AS (
    ... CASE event_name
        WHEN 'PRIME DAY' THEN 1
        WHEN 'BLACK FRIDAY' THEN 2
        WHEN 'CYBER MONDAY' THEN 3
        WHEN 'BOXING WEEK' THEN 4
        WHEN 'BSS' THEN 5
        WHEN 'NYNY' THEN 6
        ELSE 99
    END) as event_priority_order
```

## 2. Overlap Handling
* Groups overlapping events by:
    * ASIN (product level)
    * Event month (temporal grouping)
* Within each group, ranks events based on:
    * Priority order (1-99)
    * Start date (as tiebreaker)

## 3. Final Selection  
* Only keeps the highest priority event (rank = 1) for each ASIN/month combination
* If two events have same priority, earlier start date wins

```
where event_rank = 1
```

Example:
```
ASIN123 in July has:
- Prime Day (priority 1)
- Flash Sale (priority 99)

---> Prime Day will be kept, Flash Sale dropped
```
    
## 4. Conclusion
This approach ensures that:
* Each product (ASIN) has at most one event per month
* More important events (like Prime Day) take precedence
* Clear handling of conflicts through priority system