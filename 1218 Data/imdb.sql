-- MySQL Script generated by MySQL Workbench
-- Thu Jan 18 19:54:36 2024
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema IMDB
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `IMDB` ;

-- -----------------------------------------------------
-- Schema IMDB
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `IMDB` DEFAULT CHARACTER SET utf8 ;
USE `IMDB` ;

-- -----------------------------------------------------
-- Table `IMDB`.`title_basics`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IMDB`.`title_basics` ;

CREATE TABLE IF NOT EXISTS `IMDB`.`title_basics` (
  `tconst` INT NOT NULL,
  `primary_title` VARCHAR(55) NULL,
  `start_year` DATETIME NULL,
  `runtime` DECIMAL(10,2) NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IMDB`.`ratings`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IMDB`.`ratings` ;

CREATE TABLE IF NOT EXISTS `IMDB`.`ratings` (
  `tconst` INT NOT NULL,
  `average_rating` VARCHAR(55) NULL,
  `number_of_votes` INT NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IMDB`.`genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IMDB`.`genres` ;

CREATE TABLE IF NOT EXISTS `IMDB`.`genres` (
  `genre_id` INT NOT NULL,
  `genre_name` VARCHAR(55) NULL,
  PRIMARY KEY (`genre_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IMDB`.`title_genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IMDB`.`title_genres` ;

CREATE TABLE IF NOT EXISTS `IMDB`.`title_genres` (
  `genre_id` INT NOT NULL,
  `tconst` INT NOT NULL,
  PRIMARY KEY (`genre_id`, `tconst`))
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
